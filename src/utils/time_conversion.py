import copy
from datetime import datetime


def unix_to_datetime(data_dict):
    """
    Convert Unix timestamps to datetime objects in a patient data dictionary.

    Args:
        data_dict: Dictionary with patient data containing episodes with timestamp fields

    Returns:
        Dictionary with timestamps converted to datetime objects
    """
    result = copy.deepcopy(data_dict)

    if "episodes" not in result:
        return result

    for episode in result["episodes"]:
        # Convert startTime
        if "startTime" in episode:
            episode["startTime"] = datetime.fromtimestamp(episode["startTime"] / 1000)

        # Convert timestamps in bloodGlucose tuples
        if "bloodGlucose" in episode:
            for i, tuple_data in enumerate(episode["bloodGlucose"]):
                if len(tuple_data) >= 1:
                    episode["bloodGlucose"][i] = [
                        datetime.fromtimestamp(tuple_data[0] / 1000),
                        *tuple_data[1:],
                    ]

        # Convert timestamps in insulinInfusion tuples
        if "insulinInfusion" in episode:
            for i, tuple_data in enumerate(episode["insulinInfusion"]):
                if len(tuple_data) >= 1:
                    episode["insulinInfusion"][i] = [
                        datetime.fromtimestamp(tuple_data[0] / 1000),
                        *tuple_data[1:],
                    ]

        if "insulinBolus" in episode:
            for i, tuple_data in enumerate(episode["insulinBolus"]):
                if len(tuple_data) >= 1:
                    episode["insulinBolus"][i] = [
                        datetime.fromtimestamp(tuple_data[0] / 1000),
                        *tuple_data[1:],
                    ]

        # Convert timestamps in nutritionInfusion tuples
        if "nutritionInfusion" in episode:
            for i, tuple_data in enumerate(episode["nutritionInfusion"]):
                if len(tuple_data) >= 1:
                    episode["nutritionInfusion"][i] = [
                        datetime.fromtimestamp(tuple_data[0] / 1000),
                        *tuple_data[1:],
                    ]

        # Convert timestamps in nutritionBolus tuples
        if "nutritionBolus" in episode:
            for i, tuple_data in enumerate(episode["nutritionBolus"]):
                if len(tuple_data) >= 1:
                    episode["nutritionBolus"][i] = [
                        datetime.fromtimestamp(tuple_data[0] / 1000),
                        *tuple_data[1:],
                    ]

    return result
