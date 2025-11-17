import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)

from utils.generic_utils import load_json_file, get_json_files, save_json
from utils.data_helpers import extract_prediction_info, calculate_interval_midpoint
from STAR_model import STARWrapper


def process_single_patient(filepath: str, model: STARWrapper) -> dict:
    """
    Process one patient file.

    Args:
        filepath (str): Path to patient JSON file.
        model (STARWrapper): STAR API wrapper instance.

    Returns:
        dict: Result dictionary with predictions and validation.
    """

    try:
        patient_data = load_json_file(filepath)
        pred_time, actual_value = extract_prediction_info(patient_data)

        pred_interval = model.predict(
            patient_data=patient_data, prediction_time=pred_time
        )
        is_in_range = model.validate_prediction(
            interval=pred_interval, ground_truth=actual_value
        )

        interval_center = calculate_interval_midpoint(pred_interval)

        return {
            "file_name": filepath,
            "ground_truth": actual_value,
            "BG5TH": pred_interval["BG5TH"],
            "BG95TH": pred_interval["BG95TH"],
            "interval_center": interval_center,
            "is_in_range": is_in_range,
            "success": True,
            "error_message": None,
        }
    except Exception as e:
        return {
            "file_name": filepath,
            "ground_truth": None,
            "BG5TH": None,
            "BG95TH": None,
            "interval_center": None,
            "is_in_range": None,
            "success": False,
            "error_message": str(e),
        }


def parallel_predict_patients(data_path: str, max_workers: int = 10) -> pd.DataFrame:
    """
    Process patient files in parallel and return results as DataFrame.

    Args:
        data_path (str): Path to directory containing patient JSON files.
        max_workers (int): Number of parallel workers (default: 10).

    Returns:
        pd.DataFrame: Results with prediction intervals and validation.
    """

    patient_files = get_json_files(data_path)
    model_wrapper = STARWrapper()

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_patient, file, model_wrapper)
            for file in patient_files
        ]

        for future in tqdm(
            as_completed(futures), total=len(patient_files), desc="Processing"
        ):
            results.append(future.result())

    return pd.DataFrame(results)


def calculate_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate coverage rate, MAE, RMSE, and MAPE from results DataFrame.

    Args:
        df (pd.DataFrame): Results DataFrame with predictions.

    Returns:
        Dict[str, float]: Dictionary with coverage_rate, mae, rmse, mape.
    """

    # Filter only successful predictions
    df_success = df[df["success"] == True].copy()

    # Coverage Rate: % of ground truth values inside the predicted interval
    coverage_rate = df_success["is_in_range"].mean()

    # Point prediction errors using interval center
    y_true = df_success["ground_truth"].values
    y_pred = df_success["interval_center"].values

    # MAE: Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)

    # RMSE: Root Mean Squared Error
    rmse = root_mean_squared_error(y_true, y_pred)

    # MAPE: Mean Absolute Percentage Error
    mape = mean_absolute_percentage_error(y_true, y_pred)

    return {"coverage_rate": coverage_rate, "mae": mae, "rmse": rmse, "mape": mape}


def create_adversarial_evaluation_report(
    rwd_metrics: Dict[str, float],
    synth_metrics: Dict[str, float],
) -> Dict[str, Any]:
    """
    Create adversarial evaluation report comparing RWD and synthetic data metrics.

    Args:
        rwd_metrics (Dict[str, float]): Metrics from real-world data.
        synth_metrics (Dict[str, float]): Metrics from synthetic data.

    Returns:
        Dict[str, Any]: Adversarial evaluation report.
    """

    report = {
        "information": (
            "Adversarial evaluation comparing STAR model performance on real-world data (RWD) "
            "versus synthetic data. The evaluation assesses whether the model trained on real data "
            "generalizes similarly to synthetic data, indicating synthetic data quality and realism. "
            "\n\n"
            "Metrics:\n"
            "- Coverage Rate: Percentage of ground truth values falling within predicted intervals "
            "(BG5TH to BG95TH). Higher values indicate better calibrated predictions.\n"
            "- MAE (Mean Absolute Error): Average absolute difference between predicted interval midpoints "
            "and ground truth values. Lower is better.\n"
            "- RMSE (Root Mean Squared Error): Square root of average squared errors, penalizing larger errors "
            "more heavily. Lower is better.\n"
            "- MAPE (Mean Absolute Percentage Error): Average absolute percentage error, useful for "
            "comparing performance across different scales. Lower is better.\n"
            "\n"
            "Interpretation:\n"
            "Small differences between RWD and synthetic metrics suggest the synthetic data captures real-world "
            "patterns well and can be used as a valid substitute for model evaluation. Large differences "
            "indicate distribution mismatch and potential limitations in synthetic data utility."
        ),
        "Coverage Rate": {
            "rwd": f"{round(rwd_metrics["coverage_rate"]*100, 2)}pp",
            "synthetic": f"{round(synth_metrics["coverage_rate"]*100, 2)}pp",
            "difference": f"{round(abs(rwd_metrics['coverage_rate'] - synth_metrics['coverage_rate'])*100, 2)}pp",
        },
        "MAE": {
            "rwd": round(rwd_metrics["mae"], 4),
            "synthetic": round(synth_metrics["mae"], 4),
            "difference": f"{abs(rwd_metrics['mae'] - synth_metrics['mae']):.4f}",
        },
        "RMSE": {
            "rwd": round(rwd_metrics["rmse"], 4),
            "synthetic": round(synth_metrics["rmse"], 4),
            "difference": f"{abs(rwd_metrics['rmse'] - synth_metrics['rmse']):.4f}",
        },
        "MAPE": {
            "rwd": f"{round(rwd_metrics["mape"]*100, 2)}pp",
            "synthetic": f"{round(synth_metrics["mape"]*100, 2)}pp",
            "difference": f"{round(abs(rwd_metrics['mape'] - synth_metrics['mape'])*100, 2)}pp",
        },
    }

    return report


def do_adversarial_evaluation(synth_dir, rwd_dir, output_path):

    # Predictions for synth and rwd data
    synth_predictions = parallel_predict_patients(data_path=synth_dir)
    rwd_predictions = parallel_predict_patients(data_path=rwd_dir)

    # Calculate metrics
    synth_metrics = calculate_metrics(synth_predictions)
    rwd_metrics = calculate_metrics(rwd_predictions)

    # Combine metrics results
    adversarial_report = create_adversarial_evaluation_report(
        rwd_metrics=rwd_metrics, synth_metrics=synth_metrics
    )

    # Store results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_json(data=adversarial_report, filepath=output_path)

    print(f"Adversarial Evaluation Completed. Results saved to {output_path}")


def main() -> None:
    """Main entry point for adversarial evaluation of STAR synthetic vs real-world data.

    Parses command line arguments and executes adversarial evaluation comparing
    model performance on synthetic and real-world datasets using api model.
    """

    parser = argparse.ArgumentParser(
        description="Run adversarial evaluation for STAR synthetic data"
    )
    parser.add_argument(
        "--synth_dir", required=True, help="Path to tabular synthetic data directory"
    )

    parser.add_argument(
        "--rwd_dir",
        required=True,
        help="Path to tabular RWD data directory",
    )

    parser.add_argument(
        "--output",
        default="output/adversarial_evaluation_results.json",
        help="Output JSON file path",
    )

    args = parser.parse_args()

    # Execute adversarial evaluation
    do_adversarial_evaluation(
        synth_dir=args.synth_dir,
        rwd_dir=args.rwd_dir,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
