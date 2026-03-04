"""
Inference engine for anomaly detection.
"""
import numpy as np
import torch
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .data_loader import VesselDataLoader, MODEL_FEATURES, VARIABLE_GROUPS
from .model import TransformerAutoencoder, load_model
from .cbm import sliding_window_average


# Severity thresholds (based on percentile of reconstruction error)
SEVERITY_THRESHOLDS = {
    'healthy': 0.0,      # < 90th percentile
    'caution': 0.90,     # 90-95th percentile
    'warning': 0.95,     # 95-99th percentile
    'critical': 0.99     # > 99th percentile
}


@dataclass
class AnomalyResult:
    """Result of anomaly detection for a single window."""
    timestamp: datetime
    anomaly_score: float
    is_anomaly: bool
    severity: str
    reconstruction: np.ndarray
    feature_errors: Dict[str, float] = field(default_factory=dict)
    top_contributors: List[Tuple[str, float]] = field(default_factory=list)


class AnomalyDetector:
    """Anomaly detection using trained Transformer Autoencoder."""

    def __init__(
        self,
        model_path: str,
        data_loader: VesselDataLoader,
        device: Optional[torch.device] = None
    ):
        self.data_loader = data_loader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        self.model, self.metadata = load_model(model_path, self.device)
        self.threshold = self.metadata.get('threshold', 0.1)

        # Compute severity thresholds from the base threshold
        self._compute_severity_thresholds()

        # Cache for anomaly history
        self._anomaly_history: List[AnomalyResult] = []

    def _compute_severity_thresholds(self):
        """Compute severity thresholds based on the anomaly threshold."""
        # The model threshold is at 95th percentile
        # We estimate other thresholds relative to it
        self.severity_levels = {
            'healthy': 0.0,
            'caution': self.threshold * 0.8,    # ~90th percentile
            'warning': self.threshold,          # 95th percentile
            'critical': self.threshold * 2.0   # ~99th percentile
        }

    def _get_severity(self, score: float) -> str:
        """Get severity level from anomaly score."""
        if score >= self.severity_levels['critical']:
            return 'critical'
        elif score >= self.severity_levels['warning']:
            return 'warning'
        elif score >= self.severity_levels['caution']:
            return 'caution'
        else:
            return 'healthy'

    def detect(self, window: np.ndarray, timestamp: Optional[datetime] = None) -> AnomalyResult:
        """
        Detect anomaly in a single window.

        Args:
            window: (window_size, n_features) normalized feature array
            timestamp: Timestamp for this window

        Returns:
            AnomalyResult with detection details
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Ensure correct shape
        if window.ndim == 2:
            window = window[np.newaxis, ...]  # Add batch dimension

        # Convert to tensor
        x = torch.FloatTensor(window).to(self.device)

        with torch.no_grad():
            # Get reconstruction
            reconstruction = self.model(x)

            # Compute per-feature errors
            feature_errors = self.model.get_feature_errors(x)

            # Compute overall anomaly score
            anomaly_scores = self.model.compute_anomaly_score(x)

        # Convert to numpy
        reconstruction = reconstruction.cpu().numpy()[0]  # Remove batch dim
        feature_errors = feature_errors.cpu().numpy()[0]
        anomaly_scores = anomaly_scores.cpu().numpy()[0]

        # Average score over the window
        avg_score = float(anomaly_scores.mean())

        # Get per-feature error contribution (averaged over time)
        feature_error_dict = {}
        for i, name in enumerate(MODEL_FEATURES):
            feature_error_dict[name] = float(feature_errors[:, i].mean())

        # Get top contributors
        sorted_features = sorted(feature_error_dict.items(), key=lambda x: x[1], reverse=True)
        top_contributors = sorted_features[:5]

        # Determine if anomaly and severity
        is_anomaly = avg_score > self.threshold
        severity = self._get_severity(avg_score)

        result = AnomalyResult(
            timestamp=timestamp,
            anomaly_score=avg_score,
            is_anomaly=is_anomaly,
            severity=severity,
            reconstruction=reconstruction,
            feature_errors=feature_error_dict,
            top_contributors=top_contributors
        )

        # Add to history if anomaly
        if is_anomaly:
            self._anomaly_history.append(result)
            # Keep last 1000 anomalies
            if len(self._anomaly_history) > 1000:
                self._anomaly_history = self._anomaly_history[-1000:]

        return result

    def detect_batch(self, windows: np.ndarray, timestamps: Optional[List[datetime]] = None) -> List[AnomalyResult]:
        """
        Detect anomalies in multiple windows.

        Args:
            windows: (n_windows, window_size, n_features) array
            timestamps: List of timestamps for each window

        Returns:
            List of AnomalyResult
        """
        if timestamps is None:
            timestamps = [datetime.now() + timedelta(seconds=i*5) for i in range(len(windows))]

        results = []
        for i, window in enumerate(windows):
            result = self.detect(window, timestamps[i])
            results.append(result)

        return results

    def get_current_status(self) -> Dict:
        """
        Get current vessel status from the latest data window.

        Returns:
            Dictionary with current status information
        """
        # Get latest data
        latest_data = self.data_loader.get_latest(n_samples=120)

        # Normalize
        features = self.data_loader.normalize(latest_data.features)

        # Detect anomaly
        result = self.detect(features, timestamp=latest_data.timestamp[-1])

        # Get raw values for display
        raw_df = latest_data.raw_df.iloc[-1]

        # Compute total power
        total_power = float(raw_df.get('Bus1_Load', 0) + raw_df.get('Bus2_Load', 0))

        # Compute maneuver power
        maneuver_power = sum(
            float(raw_df.get(col, 0))
            for col in VARIABLE_GROUPS['maneuver']
            if col in raw_df.index
        )

        # Compute propulsion power
        propulsion_power = sum(
            float(raw_df.get(col, 0))
            for col in VARIABLE_GROUPS['propulsion']
            if col in raw_df.index
        )

        return {
            'timestamp': str(latest_data.timestamp[-1]),
            'anomaly_score': result.anomaly_score,
            'is_anomaly': result.is_anomaly,
            'severity': result.severity,
            'speed': float(raw_df.get('Speed', 0)),
            'latitude': float(raw_df.get('Latitude', 0)),
            'longitude': float(raw_df.get('Longitude', 0)),
            'total_power': total_power,
            'maneuver_power': maneuver_power,
            'propulsion_power': propulsion_power,
            'bus1_load': float(raw_df.get('Bus1_Load', 0)),
            'bus2_load': float(raw_df.get('Bus2_Load', 0)),
            'top_contributors': result.top_contributors,
            'feature_errors': result.feature_errors
        }

    def get_status_at_index(self, index: int) -> Dict:
        """
        Get vessel status at a specific data index.

        Args:
            index: Index into the dataset

        Returns:
            Dictionary with status information
        """
        # Get data window ending at this index
        data = self.data_loader.get_data_at_index(index, n_samples=120)

        if len(data.features) < 120:
            # Not enough data, return current status
            return self.get_current_status()

        # Normalize and detect
        features = self.data_loader.normalize(data.features)
        result = self.detect(features, timestamp=data.timestamp[-1])

        # Get raw values from the target index
        raw_df = data.raw_df.iloc[-1]

        # Compute power metrics (same as get_current_status)
        total_power = float(raw_df.get('Bus1_Load', 0) + raw_df.get('Bus2_Load', 0))
        maneuver_power = sum(
            float(raw_df.get(col, 0))
            for col in VARIABLE_GROUPS['maneuver']
            if col in raw_df.index
        )
        propulsion_power = sum(
            float(raw_df.get(col, 0))
            for col in VARIABLE_GROUPS['propulsion']
            if col in raw_df.index
        )

        return {
            'timestamp': str(data.timestamp[-1]),
            'anomaly_score': result.anomaly_score,
            'is_anomaly': result.is_anomaly,
            'severity': result.severity,
            'speed': float(raw_df.get('Speed', 0)),
            'latitude': float(raw_df.get('Latitude', 0)),
            'longitude': float(raw_df.get('Longitude', 0)),
            'total_power': total_power,
            'maneuver_power': maneuver_power,
            'propulsion_power': propulsion_power,
            'bus1_load': float(raw_df.get('Bus1_Load', 0)),
            'bus2_load': float(raw_df.get('Bus2_Load', 0)),
            'top_contributors': result.top_contributors,
            'feature_errors': result.feature_errors
        }

    def get_test_data_info(self) -> Dict:
        """Get info about test data range for slider."""
        test_start, test_end = self.data_loader.get_test_data_range()

        # Get timestamps at start and end
        df = self.data_loader._df
        start_time = df.index[test_start]
        end_time = df.index[test_end]

        return {
            'start_index': test_start,
            'end_index': test_end,
            'start_time': str(start_time),
            'end_time': str(end_time),
            'total_samples': test_end - test_start + 1
        }

    def get_variable_readings(self, group: str) -> Dict:
        """
        Get current readings for a variable group.

        Args:
            group: One of 'electrical', 'maneuver', 'propulsion', 'ship', 'coordinates'

        Returns:
            Dictionary with variable readings
        """
        latest_data = self.data_loader.get_latest(n_samples=1)
        raw_df = latest_data.raw_df.iloc[-1]

        variables = VARIABLE_GROUPS.get(group, [])
        readings = {}

        for var in variables:
            if var in raw_df.index:
                readings[var] = float(raw_df[var])

        # Compute group total if applicable (only sum actual loads, not available capacity)
        if group in ['electrical', 'maneuver', 'propulsion']:
            # Only sum variables that represent actual power usage (exclude 'Avail' capacity)
            total = sum(v for k, v in readings.items() if 'Avail' not in k)
            readings['total'] = total

        return {
            'group': group,
            'timestamp': str(latest_data.timestamp[-1]),
            'readings': readings
        }

    def get_variable_readings_at_index(self, group: str, index: int) -> Dict:
        """
        Get readings for a variable group at a specific index.

        Args:
            group: One of 'electrical', 'maneuver', 'propulsion', 'ship', 'coordinates'
            index: Index into the dataset

        Returns:
            Dictionary with variable readings at that index
        """
        data = self.data_loader.get_data_at_index(index, n_samples=1)
        raw_df = data.raw_df.iloc[-1]

        variables = VARIABLE_GROUPS.get(group, [])
        readings = {}

        for var in variables:
            if var in raw_df.index:
                readings[var] = float(raw_df[var])

        # Compute group total if applicable (only sum actual loads, not available capacity)
        if group in ['electrical', 'maneuver', 'propulsion']:
            total = sum(v for k, v in readings.items() if 'Avail' not in k)
            readings['total'] = total

        return {
            'group': group,
            'timestamp': str(data.timestamp[-1]),
            'readings': readings
        }

    def get_feature_health(self) -> Dict[str, str]:
        """
        Get health status for each feature.

        Returns:
            Dictionary mapping feature name to health status
        """
        # Get latest data
        latest_data = self.data_loader.get_latest(n_samples=120)
        features = self.data_loader.normalize(latest_data.features)

        # Detect anomaly (pass actual data timestamp)
        result = self.detect(features, timestamp=latest_data.timestamp[-1])

        # Determine health per feature
        health = {}
        for feature, error in result.feature_errors.items():
            if error > self.threshold * 2:
                health[feature] = 'critical'
            elif error > self.threshold:
                health[feature] = 'warning'
            elif error > self.threshold * 0.8:
                health[feature] = 'caution'
            else:
                health[feature] = 'healthy'

        return health

    def get_anomaly_history(self, hours: int = 24) -> List[Dict]:
        """
        Get recent anomaly detections by scanning through data.

        Args:
            hours: Number of hours to look back

        Returns:
            List of anomaly events
        """
        # Scan through recent data to find actual anomalies
        n_samples = int(hours * 3600 / 5)  # 5-second sampling
        data = self.data_loader.get_latest(n_samples=max(n_samples, 120))

        # Normalize
        features = self.data_loader.normalize(data.features)

        # Process in windows to find anomalies
        window_size = 120
        n_total = len(features)
        stride = window_size  # Non-overlapping for history scan

        anomalies = []

        for start_idx in range(0, n_total - window_size + 1, stride):
            end_idx = start_idx + window_size
            window = features[start_idx:end_idx]
            timestamp = data.timestamp[end_idx - 1]  # Use end of window timestamp

            # Detect anomaly
            result = self.detect(window, timestamp)

            if result.is_anomaly:
                anomalies.append({
                    'timestamp': str(result.timestamp),
                    'anomaly_score': result.anomaly_score,
                    'severity': result.severity,
                    'top_contributors': result.top_contributors
                })

        return anomalies

    def analyze_anomaly(self, timestamp_str: str) -> Dict:
        """
        Analyze a specific anomaly event.

        Args:
            timestamp_str: ISO format timestamp

        Returns:
            Detailed analysis of the anomaly
        """
        # Parse timestamp
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
        except ValueError:
            return {'error': 'Invalid timestamp format'}

        # Get data around that timestamp
        start = timestamp - timedelta(minutes=10)
        end = timestamp + timedelta(minutes=10)

        try:
            data = self.data_loader.get_time_range(start, end)
        except Exception as e:
            return {'error': f'Could not retrieve data: {str(e)}'}

        if len(data.features) < 120:
            return {'error': 'Insufficient data for analysis'}

        # Normalize
        features = self.data_loader.normalize(data.features)

        # Get detection result
        result = self.detect(features, timestamp)

        # Find the most anomalous segment
        x = torch.FloatTensor(features[np.newaxis, ...]).to(self.device)
        with torch.no_grad():
            scores = self.model.compute_anomaly_score(x).cpu().numpy()[0]

        peak_idx = int(np.argmax(scores))
        peak_time = data.timestamp[peak_idx]

        return {
            'requested_timestamp': timestamp_str,
            'peak_anomaly_time': str(peak_time),
            'peak_anomaly_score': float(scores[peak_idx]),
            'average_score': float(scores.mean()),
            'severity': result.severity,
            'top_contributors': result.top_contributors,
            'analysis': self._generate_analysis(result)
        }

    def _generate_analysis(self, result: AnomalyResult) -> str:
        """Generate human-readable analysis of anomaly."""
        if not result.is_anomaly:
            return "No significant anomaly detected in this window."

        analysis = []

        if result.severity == 'critical':
            analysis.append("CRITICAL: Significant deviation from normal operation detected.")
        elif result.severity == 'warning':
            analysis.append("WARNING: Notable deviation from expected patterns.")
        else:
            analysis.append("CAUTION: Minor deviation observed.")

        # Identify contributing factors
        if result.top_contributors:
            top_vars = [f"{name} ({error:.4f})" for name, error in result.top_contributors[:3]]
            analysis.append(f"Top contributing variables: {', '.join(top_vars)}")

            # Group analysis
            groups_affected = set()
            for var, _ in result.top_contributors[:3]:
                for group, vars in VARIABLE_GROUPS.items():
                    if var in vars:
                        groups_affected.add(group)

            if groups_affected:
                analysis.append(f"Affected systems: {', '.join(groups_affected)}")

        return " ".join(analysis)

    def get_reconstruction_at_index(
        self,
        variable: str,
        index: int,
        hours: float = 1.0
    ) -> Dict:
        """
        Get reconstruction comparison centered around a specific index.

        Args:
            variable: Variable name
            index: Center index in the dataset
            hours: Hours of data to retrieve (total window size)

        Returns:
            Dictionary with time series data centered around index
        """
        # Get variable index first
        if variable not in MODEL_FEATURES:
            return {'error': f'Variable {variable} not found'}
        var_idx = MODEL_FEATURES.index(variable)

        n_samples = int(hours * 3600 / 5)  # 5-second sampling

        # Get data window centered around the index
        half_window = n_samples // 2
        start_idx = max(0, index - half_window)
        end_idx = min(len(self.data_loader._df), index + half_window)
        actual_samples = end_idx - start_idx

        # Get data at this range
        data = self.data_loader.get_data_at_index(end_idx, n_samples=actual_samples)

        if len(data.features) < 120:
            # Not enough data, fall back to latest
            return self.get_reconstruction_comparison(variable, hours)

        # Use the common reconstruction logic
        return self._compute_reconstruction(data, var_idx, variable)

    def _compute_reconstruction(self, data, var_idx: int, variable: str) -> Dict:
        """
        Common reconstruction computation logic.

        Args:
            data: DataWindow with features and timestamps
            var_idx: Index of the variable in MODEL_FEATURES
            variable: Variable name for the result

        Returns:
            Dictionary with reconstruction data
        """
        # Normalize
        features = self.data_loader.normalize(data.features)

        # Model expects windows of 120 samples (max_seq_len)
        window_size = 120
        n_total = len(features)

        # Process in sliding windows and stitch together
        all_actual = []
        all_reconstructed = []
        all_timestamps = []

        stride = window_size // 2  # 50% overlap

        for start_idx in range(0, n_total - window_size + 1, stride):
            end_idx = start_idx + window_size
            window = features[start_idx:end_idx]

            # Get reconstruction for this window
            x = torch.FloatTensor(window[np.newaxis, ...]).to(self.device)
            with torch.no_grad():
                reconstruction = self.model(x).cpu().numpy()[0]

            # Denormalize
            actual_window = self.data_loader.denormalize(window)
            reconstructed_window = self.data_loader.denormalize(reconstruction)

            # For first window, take all values
            # For subsequent windows, only take the second half to avoid overlap
            if start_idx == 0:
                all_actual.extend(actual_window[:, var_idx].tolist())
                all_reconstructed.extend(reconstructed_window[:, var_idx].tolist())
                all_timestamps.extend([str(t) for t in data.timestamp[start_idx:end_idx]])
            else:
                # Only add the new (non-overlapping) portion
                half = window_size // 2
                all_actual.extend(actual_window[half:, var_idx].tolist())
                all_reconstructed.extend(reconstructed_window[half:, var_idx].tolist())
                all_timestamps.extend([str(t) for t in data.timestamp[start_idx + half:end_idx]])

        # Calculate error
        error = [a - r for a, r in zip(all_actual, all_reconstructed)]

        return {
            'variable': variable,
            'timestamps': all_timestamps,
            'actual': all_actual,
            'reconstructed': all_reconstructed,
            'error': error
        }

    def get_reconstruction_comparison(
        self,
        variable: str,
        hours: float = 1.0
    ) -> Dict:
        """
        Get actual vs reconstructed values for a variable.

        Args:
            variable: Variable name
            hours: Hours of data to retrieve

        Returns:
            Dictionary with time series data
        """
        # Get variable index first
        if variable not in MODEL_FEATURES:
            return {'error': f'Variable {variable} not found'}
        var_idx = MODEL_FEATURES.index(variable)

        # Get recent data
        n_samples = int(hours * 3600 / 5)  # 5-second sampling
        data = self.data_loader.get_latest(n_samples=max(n_samples, 120))

        return self._compute_reconstruction(data, var_idx, variable)

    def get_all_features_reconstruction_at_index(self, index: int, hours: float = 1.0) -> Dict:
        """
        Get reconstruction data for all features centered around a specific index.

        Args:
            index: Center index in the dataset
            hours: Hours of data to retrieve (total window size)

        Returns:
            Same as get_all_features_reconstruction but centered around index
        """
        n_samples = int(hours * 3600 / 5)  # 5-second sampling

        # Get data window centered around the index
        half_window = n_samples // 2
        start_idx = max(0, index - half_window)
        end_idx = min(len(self.data_loader._df), index + half_window)
        actual_samples = end_idx - start_idx

        # Get data at this range
        data = self.data_loader.get_data_at_index(end_idx, n_samples=actual_samples)

        if len(data.features) < 120:
            # Not enough data, fall back to latest
            return self.get_all_features_reconstruction(hours)

        return self._compute_all_features_reconstruction(data)

    # ------------------------------------------------------------------
    # Trend prediction
    # ------------------------------------------------------------------

    def _compute_trend_prediction(self, data, hours: float) -> Dict:
        """
        Core trend prediction logic: compute reconstruction errors over
        a data window, smooth them, fit linear regression, and extrapolate
        to the anomaly threshold.

        Args:
            data: DataWindow with features and timestamps
            hours: Hours of data analysed (for reporting)

        Returns:
            Dict with trend prediction results
        """
        features = self.data_loader.normalize(data.features)

        window_size = 120
        n_total = len(features)

        if n_total < window_size:
            return {'error': 'Insufficient data for trend prediction'}

        # Compute per-window reconstruction errors (stride=1)
        n_windows = n_total - window_size + 1
        errors = np.zeros(n_windows)

        for i in range(n_windows):
            window = features[i:i + window_size]
            x = torch.FloatTensor(window[np.newaxis, ...]).to(self.device)
            with torch.no_grad():
                scores = self.model.compute_anomaly_score(x)
            errors[i] = float(scores.mean())

        # Smooth with W=50 moving average
        smoothed = sliding_window_average(errors, window_size=50)

        current_score = float(smoothed[-1])

        # Linear regression on the smoothed errors
        x_vals = np.arange(len(smoothed))
        coeffs = np.polyfit(x_vals, smoothed, 1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])

        # R-squared
        y_pred = np.polyval(coeffs, x_vals)
        ss_res = np.sum((smoothed - y_pred) ** 2)
        ss_tot = np.sum((smoothed - smoothed.mean()) ** 2)
        r_squared = float(1.0 - ss_res / (ss_tot + 1e-10))

        # Classify trend direction
        if slope > 1e-7:
            trend = 'rising'
        elif slope < -1e-7:
            trend = 'falling'
        else:
            trend = 'stable'

        # Extrapolate to threshold
        predicted_failure_sample = None
        estimated_minutes_to_threshold = None

        if slope > 1e-12 and current_score < self.threshold:
            samples_to_threshold = (self.threshold - current_score) / slope
            predicted_failure_sample = int(len(smoothed) + samples_to_threshold)
            # 5-second sampling interval
            estimated_minutes_to_threshold = round(samples_to_threshold * 5 / 60, 1)

        return {
            'current_score': round(current_score, 6),
            'trend': trend,
            'slope': round(slope, 8),
            'r_squared': round(r_squared, 4),
            'predicted_failure_sample': predicted_failure_sample,
            'estimated_minutes_to_threshold': estimated_minutes_to_threshold,
            'threshold': round(self.threshold, 6),
            'hours_analyzed': hours,
            'data_points': len(smoothed),
        }

    def get_trend_prediction(self, hours: float = 2.0) -> Dict:
        """
        Analyse the reconstruction error trend and predict time to failure.

        Args:
            hours: Hours of recent data to analyse (default 2)

        Returns:
            Dict with current_score, trend, slope, r_squared,
            predicted_failure_sample, estimated_minutes_to_threshold, threshold
        """
        n_samples = int(hours * 3600 / 5)  # 5-second sampling
        data = self.data_loader.get_latest(n_samples=max(n_samples, 120))
        return self._compute_trend_prediction(data, hours)

    def get_trend_prediction_at_index(self, index: int, hours: float = 2.0) -> Dict:
        """
        Analyse the reconstruction error trend at a specific data index.

        Args:
            index: Index into the dataset
            hours: Hours of data to analyse (default 2)

        Returns:
            Dict with trend prediction results
        """
        n_samples = int(hours * 3600 / 5)
        data = self.data_loader.get_data_at_index(index, n_samples=max(n_samples, 120))
        return self._compute_trend_prediction(data, hours)

    def _compute_all_features_reconstruction(self, data) -> Dict:
        """
        Common reconstruction computation for all features.

        Args:
            data: DataWindow with features and timestamps

        Returns:
            Dictionary with reconstruction data for all features
        """
        # Normalize
        features = self.data_loader.normalize(data.features)

        # Model expects windows of 120 samples (max_seq_len)
        window_size = 120
        n_total = len(features)

        # Process in sliding windows and stitch together
        all_actual = []
        all_reconstructed = []
        all_actual_norm = []
        all_reconstructed_norm = []
        all_timestamps = []

        stride = window_size // 2  # 50% overlap

        for start_idx in range(0, n_total - window_size + 1, stride):
            end_idx = start_idx + window_size
            window = features[start_idx:end_idx]

            # Get reconstruction for this window
            x = torch.FloatTensor(window[np.newaxis, ...]).to(self.device)
            with torch.no_grad():
                reconstruction = self.model(x).cpu().numpy()[0]

            # Denormalize for display
            actual_window = self.data_loader.denormalize(window)
            reconstructed_window = self.data_loader.denormalize(reconstruction)

            # For first window, take all values
            # For subsequent windows, only take the second half to avoid overlap
            if start_idx == 0:
                all_actual.append(actual_window)
                all_reconstructed.append(reconstructed_window)
                all_actual_norm.append(window)
                all_reconstructed_norm.append(reconstruction)
                all_timestamps.extend([str(t) for t in data.timestamp[start_idx:end_idx]])
            else:
                # Only add the new (non-overlapping) portion
                half = window_size // 2
                all_actual.append(actual_window[half:])
                all_reconstructed.append(reconstructed_window[half:])
                all_actual_norm.append(window[half:])
                all_reconstructed_norm.append(reconstruction[half:])
                all_timestamps.extend([str(t) for t in data.timestamp[start_idx + half:end_idx]])

        # Stack arrays
        actual_array = np.vstack(all_actual)
        reconstructed_array = np.vstack(all_reconstructed)
        actual_norm_array = np.vstack(all_actual_norm)
        reconstructed_norm_array = np.vstack(all_reconstructed_norm)

        # Calculate per-feature errors (absolute difference)
        errors_array = np.abs(actual_array - reconstructed_array)
        errors_norm_array = np.abs(actual_norm_array - reconstructed_norm_array)

        # Calculate total error per timestep (sum across features)
        total_error = errors_array.sum(axis=1).tolist()

        return {
            'timestamps': all_timestamps,
            'actual': actual_array,
            'reconstructed': reconstructed_array,
            'errors': errors_array,
            'errors_normalized': errors_norm_array,
            'total_error': total_error,
            'feature_names': MODEL_FEATURES
        }

    def get_all_features_reconstruction(self, hours: float = 1.0) -> Dict:
        """
        Get reconstruction data for all features.

        Args:
            hours: Hours of data to retrieve

        Returns:
            Dictionary with:
            - timestamps: list of timestamp strings
            - actual: np.array (n_timesteps, 16) - denormalized
            - reconstructed: np.array (n_timesteps, 16) - denormalized
            - errors: np.array (n_timesteps, 16) - denormalized per-feature errors
            - errors_normalized: np.array (n_timesteps, 16) - normalized errors (for threshold comparison)
            - total_error: list - sum of denormalized errors across features per timestep
            - feature_names: list of feature names
        """
        # Get recent data
        n_samples = int(hours * 3600 / 5)  # 5-second sampling
        data = self.data_loader.get_latest(n_samples=max(n_samples, 120))

        return self._compute_all_features_reconstruction(data)
