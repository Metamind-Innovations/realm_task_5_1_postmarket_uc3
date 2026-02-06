import argparse
from typing import Dict, Any
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)

from utils.generic_utils import load_json_file, get_json_files, save_json
from utils.data_helpers import extract_prediction_info, calculate_interval_midpoint, extract_hospital_id
from STAR_model import STARDockerWrapper


def process_patients_batch(
        data_path: str,
        docker_image: str = "glucomeo",
        in_docker_run: bool = False
) -> pd.DataFrame:
    """
    Process patient files in batch and return results as DataFrame.

    :param data_path: Path to directory containing patient JSON files.
    :param docker_image: Name of the Docker image containing the STAR model.
    :param in_docker_run: Indicates if script runs inside the Docker container.
    :return: DataFrame with predictions and ground truth values.
    """
    patient_files = get_json_files(data_path)

    if not patient_files:
        raise ValueError(f"No JSON files found in directory: {data_path}")

    model_wrapper = STARDockerWrapper(
        docker_image=docker_image,
        in_docker_run=in_docker_run
    )

    predictions_df = model_wrapper.predict_batch(patient_files)
    predictions_dict = predictions_df.set_index("hospitalID").to_dict("index")

    results = []
    for filepath in patient_files:
        try:
            patient_data = load_json_file(filepath)
            hospital_id = extract_hospital_id(patient_data)
            pred_time, actual_value = extract_prediction_info(patient_data)

            # Match prediction by hospitalID
            if hospital_id not in predictions_dict:
                raise ValueError(f"No prediction found for hospitalID: {hospital_id}")

            prediction = predictions_dict[hospital_id]
            bg5th = prediction["BG5TH"]
            bg95th = prediction["BG95TH"]

            interval_center = calculate_interval_midpoint({
                "BG5TH": bg5th,
                "BG95TH": bg95th
            })

            results.append({
                "file_name": Path(filepath).name,
                "hospital_id": hospital_id,
                "ground_truth": actual_value,
                "BG5TH": bg5th,
                "BG95TH": bg95th,
                "interval_center": interval_center,
                "success": True,
                "error_message": None,
            })
        except Exception as e:
            results.append({
                "file_name": Path(filepath).name,
                "hospital_id": None,
                "ground_truth": None,
                "BG5TH": None,
                "BG95TH": None,
                "interval_center": None,
                "success": False,
                "error_message": str(e),
            })

    results_df = pd.DataFrame(results)

    successful_results = results_df[results_df["success"]].copy()
    if not successful_results.empty:
        ground_truth_series = successful_results["ground_truth"]
        predictions_for_validation = successful_results[["BG5TH", "BG95TH"]]

        is_in_range = model_wrapper.validate_predictions(
            predictions=predictions_for_validation,
            ground_truth=ground_truth_series
        )

        results_df.loc[results_df["success"], "is_in_range"] = is_in_range.values

    return results_df


def calculate_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate coverage rate, MAE, RMSE, and MAPE from results DataFrame.

    :param df: Results DataFrame with predictions.
    :return: Dictionary with coverage_rate, mae, rmse, mape.
    """
    df_success = df[df["success"] == True].copy()

    coverage_rate = df_success["is_in_range"].mean()

    y_true = df_success["ground_truth"].values
    y_pred = df_success["interval_center"].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    return {"coverage_rate": coverage_rate, "mae": mae, "rmse": rmse, "mape": mape}


def create_adversarial_evaluation_report(
        rwd_metrics: Dict[str, float],
        synth_metrics: Dict[str, float],
) -> Dict[str, Any]:
    """
    Create adversarial evaluation report comparing RWD and synthetic data metrics.

    :param rwd_metrics: Metrics from real-world data.
    :param synth_metrics: Metrics from synthetic data.
    :return: Adversarial evaluation report.
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
            "rwd": f"{round(rwd_metrics['coverage_rate'] * 100, 2)}pp",
            "synthetic": f"{round(synth_metrics['coverage_rate'] * 100, 2)}pp",
            "difference": f"{round(abs(rwd_metrics['coverage_rate'] - synth_metrics['coverage_rate']) * 100, 2)}pp",
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
            "rwd": f"{round(rwd_metrics['mape'] * 100, 2)}pp",
            "synthetic": f"{round(synth_metrics['mape'] * 100, 2)}pp",
            "difference": f"{round(abs(rwd_metrics['mape'] - synth_metrics['mape']) * 100, 2)}pp",
        },
    }

    return report


def do_adversarial_evaluation(
        synth_dir,
        rwd_dir,
        output_path,
        docker_image: str = "glucomeo",
        in_docker_run: bool = False
):
    """
    Run adversarial evaluation comparing synthetic and real-world data.

    :param synth_dir: Path to synthetic data directory.
    :param rwd_dir: Path to real-world data directory.
    :param output_path: Output JSON file path.
    :param docker_image: Name of the Docker image containing the STAR model.
    :param in_docker_run: Indicates if script runs inside the Docker container.
    """
    print("Processing synthetic data...")
    synth_predictions = process_patients_batch(
        data_path=synth_dir,
        docker_image=docker_image,
        in_docker_run=in_docker_run
    )

    print("Processing real-world data...")
    rwd_predictions = process_patients_batch(
        data_path=rwd_dir,
        docker_image=docker_image,
        in_docker_run=in_docker_run
    )

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


def str2bool(v: str) -> bool:
    """
    Convert a string 'True' or 'False' to a Python boolean.
    """
    if v == "True":
        return True
    elif v == "False":
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected: 'True' or 'False'")


def main() -> None:
    """
    Main entry point for adversarial evaluation of STAR synthetic vs real-world data.
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
    parser.add_argument(
        "--docker_image",
        type=str,
        default="glucomeo",
        help="The docker image to run for the model",
    )
    parser.add_argument(
        "--in_docker",
        type=str2bool,
        default=False,
        help="Indicates if the script will be executed inside the container of the provided docker image",
    )

    args = parser.parse_args()

    # Execute adversarial evaluation
    do_adversarial_evaluation(
        synth_dir=args.synth_dir,
        rwd_dir=args.rwd_dir,
        output_path=args.output,
        docker_image=args.docker_image,
        in_docker_run=args.in_docker,
    )


if __name__ == "__main__":
    main()
