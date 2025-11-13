from typing import Dict, Optional, Any
import requests
import time
import json


class STARWrapper:
    """
    Wrapper class for the STAR API blood glucose prediction model.
    This wrapper provides an interface to the STAR API endpoint
    for predicting blood glucose evolution ranges based on patient data.
    """

    def __init__(
        self,
        url: str = "https://demo.insilicare.com/api/star/REALM/validation",
        timeout: int = 5,
    ):
        """
        Initialize the STAR API wrapper.

        Args:
            url (str): API endpoint URL. Default is the public STAR validation endpoint.
            timeout (int): Request timeout in seconds (default: 60).
        """
        self.url = url
        self.timeout = timeout

        self.headers = {"Content-Type": "application/json"}

        # Store last predictions and responses
        self._last_response = None
        self._last_prediction_interval = None
        self._last_prediction_class = None

    def _validate_patient_data(self, patient_data: Dict[str, Any]) -> None:
        """
        Validate that patient data contains required fields.

        Args:
            patient_data (Dict[str, Any]): Patient data dictionary.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        required_fields = ["__class", "hospitalID", "updateTime", "episodes"]

        for field in required_fields:
            if field not in patient_data:
                raise ValueError(f"Missing required field in patient data: {field}")

        if not patient_data["episodes"]:
            raise ValueError("Patient data must contain at least one episode")

        # Check for required episode fields
        episode = patient_data["episodes"][0]
        required_episode_fields = [
            "bloodGlucose",
            "insulinInfusion",
            "nutritionInfusion",
        ]

        for field in required_episode_fields:
            if field not in episode:
                raise ValueError(f"Missing required field in episode: {field}")

    def _validate_prediction_time(
        self, patient_data: Dict[str, Any], prediction_time: int
    ) -> None:
        """
        Validate that prediction time is within acceptable range.

        Args:
            patient_data (Dict[str, Any]): Patient data dictionary.
            prediction_time (int): Unix epoch time in milliseconds.

        Raises:
            ValueError: If prediction time is outside valid range.
        """
        update_time = patient_data["updateTime"]
        max_time = update_time + (180 * 60 * 1000)  # updateTime + 3 hours

        if prediction_time < update_time:
            raise ValueError(
                f"Prediction time ({prediction_time}) must be >= updateTime ({update_time})"
            )

        if prediction_time > max_time:
            raise ValueError(
                f"Prediction time ({prediction_time}) must be <= updateTime + 180 minutes ({max_time})"
            )

    def predict(
        self,
        patient_data: Dict[str, Any],
        prediction_time: int,
        num_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> Dict[str, float]:
        """
        Predict blood glucose range at a specific time.

        Makes an API request to predict the blood glucose evolution range
        at the specified prediction time. Returns the 5th and 95th percentile
        bounds of the predicted range.

        Args:
            patient_data (Dict[str, Any]): Complete patient data JSON object
            prediction_time (int): Unix epoch time in milliseconds when to
                predict the blood glucose range.
            num_retries (int): Number of retry attempts on API failure (default: 3).
            retry_delay (float): Initial delay between retries in seconds (default: 2.0).
                Uses exponential backoff on subsequent retries.

        Returns:
            Dict[str, float]: Dictionary containing prediction interval with keys:
                - "BG5TH" (float): Lower bound (5th percentile) of predicted range
                - "BG95TH" (float): Upper bound (95th percentile) of predicted range

        Raises:
            ValueError: If patient data or prediction time validation fails.
            RuntimeError: If API request fails after all retry attempts.
        """
        # Validate inputs
        self._validate_patient_data(patient_data)
        self._validate_prediction_time(patient_data, prediction_time)

        # Prepare request payload
        payload = {"patient": patient_data, "predictionTime": prediction_time}

        # Make API request with retry mechanism
        for attempt in range(num_retries):
            try:
                response = requests.post(
                    self.url, headers=self.headers, json=payload, timeout=self.timeout
                )
                response.raise_for_status()

                result = response.json()

                # Validate response contains expected fields
                if "BG5TH" not in result or "BG95TH" not in result:
                    raise ValueError(
                        f"API response missing required fields. Got: {result.keys()}"
                    )

                # Store response and interval
                self._last_response = result
                self._last_prediction_interval = {
                    "BG5TH": result["BG5TH"],
                    "BG95TH": result["BG95TH"],
                }

                return self._last_prediction_interval

            except requests.exceptions.Timeout as e:
                if attempt < num_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"API request timed out after {num_retries} attempts: {e}"
                    )

            except requests.exceptions.RequestException as e:
                if attempt < num_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"API request failed after {num_retries} attempts: {e}"
                    )

            except (ValueError, KeyError, json.JSONDecodeError) as e:
                # Don't retry on data/parsing errors
                raise RuntimeError(f"Error processing API response: {e}")

    def validate_prediction(
        self,
        ground_truth: float,
    ) -> int:
        """
        Check whether ground truth value falls within the last predicted range.

        Validates if the provided ground truth blood glucose value falls within
        the last prediction interval obtained from the predict() method.
        Returns 1 if the ground truth is inside the interval (correct prediction),
        0 otherwise (incorrect prediction).

        Args:
            ground_truth (float): Actual blood glucose value to compare against
                the last predicted range.

        Returns:
            int: Binary prediction correctness indicator:
                - 1 if ground_truth is within [BG5TH, BG95TH] (correct prediction)
                - 0 if ground_truth is outside the range (incorrect prediction)

        Raises:
            RuntimeError: If predict() has not been called yet (no prediction interval available).
        """

        # Check if we have a prediction interval to validate against
        if self._last_prediction_interval is None:
            raise RuntimeError(
                "No prediction interval available. Call predict() first before validating."
            )

        # Check if ground truth is within the predicted range
        bg_5th = self._last_prediction_interval["BG5TH"]
        bg_95th = self._last_prediction_interval["BG95TH"]

        is_inside = int(bg_5th <= ground_truth <= bg_95th)

        # Store validation result
        self._last_prediction_class = {
            "BG5TH": bg_5th,
            "BG95TH": bg_95th,
            "ground_truth": ground_truth,
            "is_inside": bool(is_inside),
        }

        return is_inside

    def get_last_response(self) -> Optional[Dict[str, Any]]:
        """
        Get the raw API response from the last prediction request.

        Returns:
            Optional[Dict[str, Any]]: Complete API response dictionary from
                the most recent predict or predict_class call. Returns None
                if no predictions have been made yet.
        """

        return self._last_response

    def get__last_prediction_interval(self) -> Optional[Dict[str, float]]:
        """
        Get the prediction interval from the last request.

        Returns:
            Optional[Dict[str, float]]: Dictionary with 'BG5TH' and 'BG95TH'
                keys containing the predicted blood glucose range bounds.
                Returns None if no predictions have been made yet.
        """
        return self._last_prediction_interval

    def get_last_prediction_class(self) -> Optional[Dict[str, Any]]:
        """
        Get detailed results from the last predict_class() call.

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing:
                - "BG5TH" (float): Lower bound of predicted range
                - "BG95TH" (float): Upper bound of predicted range
                - "ground_truth" (float): Ground truth value used for comparison
                - "is_inside" (bool): Whether ground truth was inside the range
                Returns None if predict_class() has not been called yet.
        """

        return self._last_prediction_class
