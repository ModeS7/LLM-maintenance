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

        # Compute group total if applicable
        if group in ['electrical', 'maneuver', 'propulsion']:
            readings['total'] = sum(readings.values())

        return {
            'group': group,
            'timestamp': str(latest_data.timestamp[-1]),
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

        # Detect anomaly
        result = self.detect(features)

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
        Get recent anomaly detections.

        Args:
            hours: Number of hours to look back

        Returns:
            List of anomaly events
        """
        cutoff = datetime.now() - timedelta(hours=hours)

        recent_anomalies = [
            {
                'timestamp': str(r.timestamp),
                'anomaly_score': r.anomaly_score,
                'severity': r.severity,
                'top_contributors': r.top_contributors
            }
            for r in self._anomaly_history
            if r.timestamp >= cutoff
        ]

        return recent_anomalies

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
