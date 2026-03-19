"""
Core CBM (Condition-Based Maintenance) module.

All CBM logic as reusable functions. No plotting, no file I/O beyond loading joblib.
"""
import numpy as np
import torch
import joblib
from dataclasses import dataclass
from typing import Optional

from src.data_loader import MODEL_FEATURES

# ---------------------------------------------------------------------------
# Key mapping: joblib dict keys  <-->  MODEL_FEATURES names
# ---------------------------------------------------------------------------
JOBLIB_TO_MODEL_KEY_MAP = {
    'Bus1_load': 'Bus1_Load',
    'Bus1_Avail_load': 'Bus1_Avail_Load',
    'Bus2_load': 'Bus2_Load',
    'Bus2_Avail_load': 'Bus2_Avail_Load',
}

MODEL_TO_JOBLIB_KEY_MAP = {v: k for k, v in JOBLIB_TO_MODEL_KEY_MAP.items()}

# ---------------------------------------------------------------------------
# Failure scenario configurations
# ---------------------------------------------------------------------------
FAILURE_CONFIGS = {
    'slow_drift': {
        'injection_point': 5000,
        'description': 'Gradual increase of all bus loads',
    },
    'load_imbalance': {
        'injection_point': 20000,
        'description': 'Bus1 decreases while Bus2 increases',
    },
    'temporary_reduction': {
        'injection_point': 10000,
        'end_point': 20000,
        'description': 'Temporary load reduction (samples 10000-20000)',
    },
    'spikes': {
        'injection_point': 10000,
        'end_point': 10500,
        'description': 'Short burst of spikes (samples 10000-10500)',
    },
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PrognosticResult:
    slope: float
    intercept: float
    predicted_failure_sample: Optional[int]
    r_squared: float
    lookback_start: int
    lookback_end: int


@dataclass
class CBMEvaluationResult:
    failure_type: str
    raw_errors: np.ndarray
    smoothed_errors: np.ndarray
    threshold: float
    anomaly_flags: np.ndarray
    injection_point: int
    first_detection: Optional[int]
    detection_delay: Optional[int]
    prognostic: Optional[PrognosticResult]
    original_data: np.ndarray
    modified_data: np.ndarray

# ---------------------------------------------------------------------------
# Data conversion
# ---------------------------------------------------------------------------

def joblib_dict_to_array(data_dict: dict) -> np.ndarray:
    """Convert joblib dict -> (N, 16) array in MODEL_FEATURES order."""
    n_samples = len(next(iter(data_dict.values())))
    array = np.zeros((n_samples, len(MODEL_FEATURES)))

    for i, feature in enumerate(MODEL_FEATURES):
        joblib_key = MODEL_TO_JOBLIB_KEY_MAP.get(feature, feature)
        array[:, i] = np.asarray(data_dict[joblib_key], dtype=float)

    return array


def load_joblib_as_array(path: str) -> np.ndarray:
    """Load .joblib file and convert to (N, 16) array."""
    data_dict = joblib.load(path)
    return joblib_dict_to_array(data_dict)

# ---------------------------------------------------------------------------
# Failure injection (vectorised replicas of the 4 scripts)
# ---------------------------------------------------------------------------

def inject_failure(data_dict: dict, failure_type: str, scale_factor: float = 1.0,
                   injection_point: Optional[int] = None) -> dict:
    """Programmatic fault injection returning a modified *copy* of the dict.

    Args:
        scale_factor: multiplier applied to all injection coefficients and
            step sizes.  The original scripts use scale_factor=1; increase
            it when bus-load magnitudes make the default offsets too small
            relative to the model's reconstruction noise floor.
        injection_point: sample index where fault begins.  If *None*, the
            default from FAILURE_CONFIGS is used.
    """
    data = {k: np.array(v, dtype=float) for k, v in data_dict.items()}

    keys_of_interest = [
        'Bus1_load', 'Bus1_Avail_load', 'Bus2_Avail_load', 'Bus2_load',
    ]
    n = len(data[keys_of_interest[0]])
    indices = np.arange(n)
    sf = scale_factor

    config = FAILURE_CONFIGS[failure_type]
    injection = injection_point if injection_point is not None else config['injection_point']

    if failure_type == 'slow_drift':
        initial_coeffs = [50.0 * sf, 25.0 * sf, 10.0 * sf, 15.0 * sf]
        passo = 0.05 * sf

        for i, key in enumerate(keys_of_interest):
            mask = indices > injection
            n_affected = mask.sum()
            if n_affected > 0:
                steps = np.arange(n_affected)
                offsets = initial_coeffs[i] + steps * passo
                data[key][mask] += offsets

    elif failure_type == 'load_imbalance':
        initial_coeffs = [-50.0 * sf, -25.0 * sf, 10.0 * sf, 15.0 * sf]
        passo = -0.05 * sf

        for i, key in enumerate(keys_of_interest):
            mask = indices > injection
            n_affected = mask.sum()
            if n_affected > 0:
                steps = np.arange(n_affected)
                if key in ('Bus1_load', 'Bus1_Avail_load'):
                    offsets = initial_coeffs[i] + steps * passo
                else:
                    offsets = initial_coeffs[i] - steps * passo
                data[key][mask] += offsets

    elif failure_type == 'temporary_reduction':
        initial_coeffs = [-50.0 * sf, -25.0 * sf, 10.0 * sf, 15.0 * sf]
        default_duration = config.get('end_point', 20000) - config['injection_point']
        end = injection + default_duration
        passo = -0.03 * sf

        for i, key in enumerate(keys_of_interest):
            mask = (indices > injection) & (indices < end)
            n_affected = mask.sum()
            if n_affected > 0:
                steps = np.arange(n_affected)
                offsets = initial_coeffs[i] + steps * passo
                data[key][mask] += offsets

    elif failure_type == 'spikes':
        initial_coeffs = [-50.0 * sf, -25.0 * sf, 10.0 * sf, 15.0 * sf]
        default_duration = config.get('end_point', 10500) - config['injection_point']
        end = injection + default_duration
        passo = 3.0 * sf

        for i, key in enumerate(keys_of_interest):
            mask = (indices > injection) & (indices < end)
            n_affected = mask.sum()
            if n_affected > 0:
                steps = np.arange(n_affected)
                offsets = initial_coeffs[i] + steps * passo
                data[key][mask] += offsets

    else:
        raise ValueError(f"Unknown failure type: {failure_type}")

    # Clamp bus loads to zero — kW values cannot be negative
    for key in keys_of_interest:
        np.maximum(data[key], 0, out=data[key])

    return data

# ---------------------------------------------------------------------------
# CBM functions
# ---------------------------------------------------------------------------

def compute_reconstruction_errors(
    data: np.ndarray,
    model: torch.nn.Module,
    scaler,
    window_size: int = 120,
    stride: int = 1,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Batched per-window MSE errors (stride=1 by default)."""
    if device is None:
        device = next(model.parameters()).device

    scaled_data = scaler.transform(data)
    n_samples = scaled_data.shape[0]
    n_windows = (n_samples - window_size) // stride + 1

    errors = np.zeros(n_windows)

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, n_windows, batch_size):
            batch_end = min(batch_start + batch_size, n_windows)

            # Vectorised window creation
            window_starts = np.arange(batch_start, batch_end) * stride
            offsets = np.arange(window_size)
            idx = window_starts[:, None] + offsets[None, :]  # (batch, window_size)
            batch_windows = scaled_data[idx]  # (batch, window_size, n_features)

            batch_tensor = torch.from_numpy(batch_windows).float().to(device)
            scores = model.compute_anomaly_score(batch_tensor)  # (batch, seq_len)
            errors[batch_start:batch_end] = scores.mean(dim=-1).cpu().numpy()

    return errors


def calibrate_threshold(
    healthy_errors: np.ndarray,
    safety_factor: float = 1.25,
) -> float:
    """max(healthy_errors) * safety_factor per the CBM proposal."""
    return float(np.max(healthy_errors) * safety_factor)


def sliding_window_average(
    errors: np.ndarray,
    window_size: int = 50,
) -> np.ndarray:
    """Causal (backward-looking) moving average, vectorised."""
    n = len(errors)
    if n == 0:
        return errors.copy()

    cumsum = np.concatenate([[0.0], np.cumsum(errors)])
    indices = np.arange(n)
    starts = np.maximum(0, indices - window_size + 1)
    lengths = indices - starts + 1
    smoothed = (cumsum[indices + 1] - cumsum[starts]) / lengths

    return smoothed


def detect_anomalies(
    smoothed_errors: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Boolean flagging: True where smoothed error exceeds threshold."""
    return smoothed_errors > threshold


def estimate_time_to_failure(
    smoothed_errors: np.ndarray,
    threshold: float,
    lookback: int = 2000,
) -> PrognosticResult:
    """Linear regression extrapolation -> predicted failure sample."""
    n = len(smoothed_errors)
    lookback_start = max(0, n - lookback)
    segment = smoothed_errors[lookback_start:]

    x = np.arange(len(segment))
    coeffs = np.polyfit(x, segment, 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])

    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((segment - y_pred) ** 2)
    ss_tot = np.sum((segment - segment.mean()) ** 2)
    r_squared = float(1.0 - ss_res / (ss_tot + 1e-10))

    predicted_failure_sample = None
    if slope > 1e-12:
        x_cross = (threshold - intercept) / slope
        predicted_failure_sample = int(lookback_start + x_cross)

    return PrognosticResult(
        slope=slope,
        intercept=intercept,
        predicted_failure_sample=predicted_failure_sample,
        r_squared=r_squared,
        lookback_start=lookback_start,
        lookback_end=n,
    )

# ---------------------------------------------------------------------------
# End-to-end pipeline for one fault scenario
# ---------------------------------------------------------------------------

def run_cbm_evaluation(
    data_dict: dict,
    failure_type: str,
    model: torch.nn.Module,
    scaler,
    healthy_errors: np.ndarray,
    threshold: float,
    window_size: int = 120,
    smoothing_window: int = 50,
    lookback: int = 2000,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
    scale_factor: float = 1.0,
    injection_point: Optional[int] = None,
) -> CBMEvaluationResult:
    """Run the full CBM evaluation for a single fault scenario."""
    # Inject failure
    modified_dict = inject_failure(data_dict, failure_type,
                                   scale_factor=scale_factor,
                                   injection_point=injection_point)

    # Resolve actual injection point used
    actual_injection = (injection_point if injection_point is not None
                        else FAILURE_CONFIGS[failure_type]['injection_point'])

    # Convert to arrays
    original_array = joblib_dict_to_array(data_dict)
    modified_array = joblib_dict_to_array(modified_dict)

    # Compute reconstruction errors on faulty data
    raw_errors = compute_reconstruction_errors(
        modified_array, model, scaler,
        window_size=window_size, stride=1,
        batch_size=batch_size, device=device,
    )

    # Smooth
    smoothed_errors = sliding_window_average(raw_errors, smoothing_window)

    # Detect
    anomaly_flags = detect_anomalies(smoothed_errors, threshold)

    # First detection after injection
    anomaly_indices = np.where(anomaly_flags)[0]
    first_detection = None
    detection_delay = None
    if len(anomaly_indices) > 0:
        post_injection = anomaly_indices[anomaly_indices >= actual_injection]
        if len(post_injection) > 0:
            first_detection = int(post_injection[0])
            detection_delay = first_detection - actual_injection

    # Prognostic (only for sustained failures)
    prognostic = None
    if failure_type in ('slow_drift', 'load_imbalance'):
        prognostic = estimate_time_to_failure(
            smoothed_errors, threshold, lookback=lookback,
        )

    return CBMEvaluationResult(
        failure_type=failure_type,
        raw_errors=raw_errors,
        smoothed_errors=smoothed_errors,
        threshold=threshold,
        anomaly_flags=anomaly_flags,
        injection_point=actual_injection,
        first_detection=first_detection,
        detection_delay=detection_delay,
        prognostic=prognostic,
        original_data=original_array,
        modified_data=modified_array,
    )
