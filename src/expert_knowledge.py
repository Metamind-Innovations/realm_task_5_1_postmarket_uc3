from datetime import datetime, timedelta
import json
import os

from src.utils.time_conversion import unix_to_datetime


def check_blood_glucose_values(data_with_datetime, filename):
    """
    Checks whether all blood glucose values in the provided data are within the expert-defined range [1.2, 110] mmol/L.

    Args:
        data_with_datetime (dict): The data dictionary containing episodes with blood glucose measurements.
        filename (str): The name of the file being checked (used as a key in the result).

    Returns:
        dict: A dictionary with the filename as the key, containing:
            - "valid" (bool): True if all values are within range, False otherwise.
            - "invalid_values" (list): List of dicts with "timestamp" and "glucose_value" for each out-of-range value.
    """
    result = {
        filename: {
            "valid": True,
            "invalid_values": [],
        }
    }

    if "episodes" in data_with_datetime:
        for episode in data_with_datetime["episodes"]:
            if "bloodGlucose" in episode:
                for bg_entry in episode["bloodGlucose"]:
                    if len(bg_entry) >= 2:
                        timestamp = bg_entry[0]
                        glucose_value = bg_entry[1]

                        if glucose_value < 1.2 or glucose_value > 110:
                            result[filename]["valid"] = False
                            result[filename]["invalid_values"].append(
                                {
                                    "timestamp": timestamp.isoformat(),
                                    "glucose_value": glucose_value,
                                }
                            )

    return result


def check_subcataneous_insulin(data_with_datetime, filename):
    """
    Checks whether subcutaneous insulin (route=1) was administered in the 12 hours prior to the last blood glucose measurement
    in each episode of the provided data.

    Args:
        data_with_datetime (dict): The data dictionary containing episodes with insulin and blood glucose measurements.
        filename (str): The name of the file being checked (used as a key in the result).

    Returns:
        dict: A dictionary with the filename as the key, containing:
            - "valid" (bool): True if no subcutaneous insulin was administered in the 12-hour window, False otherwise.
            - "invalid_administrations" (list): List of dicts with "timestamp", "type" ("infusion" or "bolus"), and "route"
              for each subcutaneous insulin administration found in the window.
    """
    result = {
        filename: {
            "valid": True,
            "invalid_administrations": [],
        }
    }

    if "episodes" in data_with_datetime:
        for episode in data_with_datetime["episodes"]:
            # Find the last blood glucose timestamp
            last_bg_timestamp = None
            if "bloodGlucose" in episode:
                for bg_entry in episode["bloodGlucose"]:
                    if len(bg_entry) >= 2:
                        timestamp = bg_entry[0]

                        if last_bg_timestamp is None or timestamp > last_bg_timestamp:
                            last_bg_timestamp = timestamp

            if last_bg_timestamp is not None:
                window_start = last_bg_timestamp - timedelta(hours=12)
                window_end = last_bg_timestamp

                # Check insulinInfusion for subcutaneous administration (route=1)
                if "insulinInfusion" in episode:
                    for infusion_entry in episode["insulinInfusion"]:
                        if len(infusion_entry) >= 2:
                            infusion_timestamp = infusion_entry[0]
                            infusion_data = infusion_entry[1]

                            if (
                                window_start <= infusion_timestamp <= window_end
                                and isinstance(infusion_data, dict)
                                and infusion_data.get("route") == 1
                            ):
                                result[filename]["valid"] = False
                                result[filename]["invalid_administrations"].append(
                                    {
                                        "timestamp": infusion_timestamp.isoformat(),
                                        "type": "infusion",
                                        "route": infusion_data.get("route"),
                                    }
                                )

                # Check insulinBolus for subcutaneous administration (route=1)
                if "insulinBolus" in episode:
                    for bolus_entry in episode["insulinBolus"]:
                        if len(bolus_entry) >= 2:
                            bolus_timestamp = bolus_entry[0]
                            bolus_data = bolus_entry[1]

                            if (
                                window_start <= bolus_timestamp <= window_end
                                and isinstance(bolus_data, dict)
                                and bolus_data.get("route") == 1
                            ):
                                result[filename]["valid"] = False
                                result[filename]["invalid_administrations"].append(
                                    {
                                        "timestamp": bolus_timestamp.isoformat(),
                                        "type": "bolus",
                                        "route": bolus_data.get("route"),
                                    }
                                )

    return result


def do_expert_knowledge_check(synthetic_samples_path, expert_knowledge_results_path):
    """
    Performs expert knowledge checks on synthetic sample data and writes the results to a JSON file.

    This function applies two expert knowledge criteria to each JSON file in the specified directory:
        1. Checks that all blood glucose values are within the plausible human range [1.2, 110] mmol/L.
        2. Checks that no subcutaneous insulin (route=1) was administered in the 12 hours prior to the last blood glucose measurement.

    Args:
        synthetic_samples_path (str): Path to the directory containing synthetic sample JSON files.
        expert_knowledge_results_path (str): Path to the output JSON file where results will be saved.

    The output JSON will contain, for each criterion, an "information" field describing the rule and a result for each file.
    """
    expert_knowledge_results = {}

    expert_knowledge_results["criterion_1"] = {}
    expert_knowledge_results["criterion_1"]["information"] = (
        "The valid humanly plausible ranges for blood glucose are [1.2, 110] mmol/L according to Barry (2020) and Manappallil (2017)"
    )

    expert_knowledge_results["criterion_2"] = {}
    expert_knowledge_results["criterion_2"]["information"] = (
        "According to Walsh et. al. (2014), subcutaneous insulin may not have been administered in the last 6 hours prior the period considered (so 12hours before the time of evaluation)"
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

            criterion_1_result = check_blood_glucose_values(
                data_with_datetime, filename
            )
            expert_knowledge_results["criterion_1"].update(criterion_1_result)

            criterion_2_result = check_subcataneous_insulin(
                data_with_datetime, filename
            )
            expert_knowledge_results["criterion_2"].update(criterion_2_result)

    with open(expert_knowledge_results_path, "w") as f:
        json.dump(expert_knowledge_results, f)
