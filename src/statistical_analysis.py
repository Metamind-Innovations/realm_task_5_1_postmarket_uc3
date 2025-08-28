import json
import os
from datetime import datetime, timedelta

from src.utils.time_conversion import unix_to_datetime


def check_required_fields(data, filename):
    """
    Checks whether all required fields are present in each episode of the provided data.

    Args:
        data (dict): The data dictionary containing episodes to be checked.
        filename (str): The name of the file being checked (used as a key in the result).

    Returns:
        dict: A dictionary with the filename as the key, containing:
            - "valid" (bool): True if all required fields are present in every episode, False otherwise.
            - "missing_fields" (list): List of missing fields (if any) for the file.
    """
    required_fields = [
        "diabeticStatus",
        "startTime",
        "bloodGlucose",
        "insulinInfusion",
        "insulinBolus",
        "nutritionInfusion",
        "nutritionBolus",
    ]

    result = {filename: {"valid": True, "missing_fields": []}}

    if "episodes" in data:
        for episode in data["episodes"]:
            for field in required_fields:
                if field not in episode:
                    result[filename]["valid"] = False
                    if field not in result[filename]["missing_fields"]:
                        result[filename]["missing_fields"].append(field)
    else:
        result[filename]["valid"] = False
        result[filename]["missing_fields"] = ["episodes"]

    return result


def check_iv_rates_not_null(data_with_datetime, filename):
    """
    Checks that at no point in time are both insulin and nutrition IV infusion rates zero.

    Args:
        data_with_datetime (dict): The data dictionary containing episodes, where all timestamps are datetime objects.
        filename (str): The name of the file being checked (used as a key in the result).

    Returns:
        dict: A dictionary with the filename as the key, containing:
            - "valid" (bool): True if there are no periods where both infusion rates are zero, False otherwise.
            - "invalid_periods" (list): List of periods (with timestamps and rates) where both rates are zero.
    """
    result = {filename: {"valid": True, "invalid_periods": []}}

    if "episodes" in data_with_datetime:
        for episode in data_with_datetime["episodes"]:
            timeline = []

            if "insulinInfusion" in episode:
                for infusion_entry in episode["insulinInfusion"]:
                    if len(infusion_entry) >= 2:
                        timestamp = infusion_entry[0]
                        infusion_data = infusion_entry[1]
                        if isinstance(infusion_data, dict):
                            rate = infusion_data.get("rate", 0)
                            timeline.append((timestamp, "insulin", rate))

            if "nutritionInfusion" in episode:
                for nutrition_entry in episode["nutritionInfusion"]:
                    if len(nutrition_entry) >= 2:
                        timestamp = nutrition_entry[0]
                        nutrition_data = nutrition_entry[1]
                        if isinstance(nutrition_data, dict):
                            rate = nutrition_data.get("rate", 0)
                            timeline.append((timestamp, "nutrition", rate))

            timeline.sort(key=lambda x: x[0])

            current_insulin_rate = 0
            current_nutrition_rate = 0

            for timestamp, infusion_type, rate in timeline:
                if infusion_type == "insulin":
                    current_insulin_rate = rate if rate is not None else 0
                elif infusion_type == "nutrition":
                    current_nutrition_rate = rate if rate is not None else 0

                if current_insulin_rate == 0 and current_nutrition_rate == 0:
                    result[filename]["valid"] = False
                    result[filename]["invalid_periods"].append(
                        {
                            "timestamp": timestamp.isoformat(),
                            "insulin_rate": current_insulin_rate,
                            "nutrition_rate": current_nutrition_rate,
                        }
                    )

    return result


def check_diabetic_status_valid(data, filename):
    """
    Checks whether the 'diabeticStatus' field in each episode has a valid value.

    Args:
        data (dict): The data dictionary containing episodes to be checked.
        filename (str): The name of the file being checked (used as a key in the result).

    Returns:
        dict: A dictionary with the filename as the key, containing:
            - "valid" (bool): True if all diabeticStatus values are valid (0, 1, or 2), False otherwise.
            - "invalid_statuses" (list): List of invalid diabeticStatus values found in the file.
    """
    valid_statuses = [0, 1, 2]

    result = {filename: {"valid": True, "invalid_statuses": []}}

    if "episodes" in data:
        for episode in data["episodes"]:
            if "diabeticStatus" in episode:
                status = episode["diabeticStatus"]
                if status not in valid_statuses:
                    result[filename]["valid"] = False
                    result[filename]["invalid_statuses"].append(status)

    return result


def check_blood_glucose_measurements(data_with_datetime, filename):
    """
    Checks whether there are at least 3 blood glucose measurements in the last 6 hours of each episode.

    Args:
        data_with_datetime (dict): The data dictionary containing episodes, where all timestamps are datetime objects.
        filename (str): The name of the file being checked (used as a key in the result).

    Returns:
        dict: A dictionary with the filename as the key, containing:
            - "valid" (bool): True if there are at least 3 blood glucose measurements in the last 6 hours for each episode, False otherwise.
            - "measurement_counts" (list): List of dicts for each episode with the window start, window end, and count of measurements in the last 6 hours.
    """
    result = {filename: {"valid": True, "measurement_counts": []}}

    if "episodes" in data_with_datetime:
        for episode in data_with_datetime["episodes"]:
            if "bloodGlucose" in episode:
                last_bg_timestamp = None
                for bg_entry in episode["bloodGlucose"]:
                    if len(bg_entry) >= 2:
                        timestamp = bg_entry[0]
                        if last_bg_timestamp is None or timestamp > last_bg_timestamp:
                            last_bg_timestamp = timestamp

                if last_bg_timestamp is not None:
                    window_start = last_bg_timestamp - timedelta(hours=6)
                    window_end = last_bg_timestamp

                    count = 0
                    for bg_entry in episode["bloodGlucose"]:
                        if len(bg_entry) >= 2:
                            timestamp = bg_entry[0]
                            if window_start <= timestamp <= window_end:
                                count += 1

                    result[filename]["measurement_counts"].append(
                        {
                            "window_start": window_start.isoformat(),
                            "window_end": window_end.isoformat(),
                            "count": count,
                        }
                    )

                    if count < 3:
                        result[filename]["valid"] = False

    return result


def do_statistical_analysis(synthetic_samples_path, statistical_analysis_results_path):
    """
    Performs a series of statistical checks on synthetic sample files in the specified directory
    and writes the results to a JSON file.

    Args:
        synthetic_samples_path (str): Path to the directory containing synthetic sample JSON files.
        statistical_analysis_results_path (str): Path to the output JSON file where results will be saved.

    The following checks are performed for each file:
        1. Required fields exist (diabeticStatus, startTime, bloodGlucose, insulinInfusion, insulinBolus, nutritionInfusion, nutritionBolus).
        2. Both IV insulin and nutrition rates cannot be null at the same time.
        3. diabeticStatus has a valid value (0, 1, 2).
        4. At least 3 blood glucose measurements in the last 6 hours.
    """
    statistical_analysis_results = {}

    statistical_analysis_results["check_1"] = {}
    statistical_analysis_results["check_1"]["information"] = (
        "Required fields exist (diabeticStatus, startTime, bloodGlucose, insulinInfusion, insulinBolus, nutritionInfusion, nutritionBolus)"
    )

    statistical_analysis_results["check_2"] = {}
    statistical_analysis_results["check_2"]["information"] = (
        "Both IV insulin and nutrition rates cannot be null at the same time"
    )

    statistical_analysis_results["check_3"] = {}
    statistical_analysis_results["check_3"]["information"] = (
        "diabeticStatus has a valid value (0,1,2)"
    )

    statistical_analysis_results["check_4"] = {}
    statistical_analysis_results["check_4"]["information"] = (
        "At least 3 blood glucose measurements in the last 6 hours"
    )

    for filename in os.listdir(synthetic_samples_path):
        if filename.endswith(".json"):
            file_path = os.path.join(synthetic_samples_path, filename)
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = None

            if data is None:
                continue

            data_with_datetime = unix_to_datetime(data)

            check_1_result = check_required_fields(data, filename)
            statistical_analysis_results["check_1"].update(check_1_result)

            check_2_result = check_iv_rates_not_null(data_with_datetime, filename)
            statistical_analysis_results["check_2"].update(check_2_result)

            check_3_result = check_diabetic_status_valid(data, filename)
            statistical_analysis_results["check_3"].update(check_3_result)

            check_4_result = check_blood_glucose_measurements(
                data_with_datetime, filename
            )
            statistical_analysis_results["check_4"].update(check_4_result)

    with open(statistical_analysis_results_path, "w") as f:
        json.dump(statistical_analysis_results, f)
