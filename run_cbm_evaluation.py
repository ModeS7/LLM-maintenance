#!/usr/bin/env python3
"""
CBM Evaluation Pipeline — Top-level script.

Loads model/scaler/data, runs all 4 fault scenarios, generates plots and a
summary report under docs/data/.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import load_model
from src.data_loader import MODEL_FEATURES
from src.cbm import (
    joblib_dict_to_array,
    compute_reconstruction_errors,
    calibrate_threshold,
    sliding_window_average,
    run_cbm_evaluation,
    FAILURE_CONFIGS,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'data'
MODEL_PATH = PROJECT_ROOT / 'models' / 'autoencoder.pt'
SCALER_PATH = PROJECT_ROOT / 'models' / 'scaler.pkl'
DATA_PATH = PROJECT_ROOT / 'docs' / 'data' / 'variable_of_interest_for_PCC.joblib'

FAULT_TYPES = ['slow_drift', 'load_imbalance', 'temporary_reduction', 'spikes']

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _contiguous_regions(flags: np.ndarray):
    """Return list of (start, end) pairs for contiguous True regions."""
    diff = np.diff(flags.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if flags[0]:
        starts = np.concatenate([[0], starts])
    if flags[-1]:
        ends = np.concatenate([ends, [len(flags)]])
    return list(zip(starts.tolist(), ends.tolist()))


def plot_healthy_baseline(healthy_errors, threshold):
    smoothed = sliding_window_average(healthy_errors, 50)
    x = np.arange(len(healthy_errors))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, healthy_errors, color='lightblue', alpha=0.4, linewidth=0.5, label='Raw error')
    ax.plot(x, smoothed, color='blue', linewidth=1.0, label='Smoothed error (w=50)')
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5,
               label=f'CBM Threshold ({threshold:.4f})')
    ax.set_title('Healthy Baseline: Reconstruction Errors')
    ax.set_xlabel('Sample')
    ax.set_ylabel('MSE')
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'healthy_baseline.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_reconstruction_error(result):
    x = np.arange(len(result.raw_errors))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, result.raw_errors, color='lightblue', alpha=0.4, linewidth=0.5,
            label='Raw error')
    ax.plot(x, result.smoothed_errors, color='blue', linewidth=1.0,
            label='Smoothed error (w=50)')
    ax.axhline(y=result.threshold, color='red', linestyle='--', linewidth=1.5,
               label=f'Threshold ({result.threshold:.4f})')
    ax.axvline(x=result.injection_point, color='green', linestyle='--', linewidth=1.5,
               label=f'Injection (sample {result.injection_point})')

    # Shade anomaly regions
    if np.any(result.anomaly_flags):
        for s, e in _contiguous_regions(result.anomaly_flags):
            ax.axvspan(s, e, color='red', alpha=0.12)

    if result.first_detection is not None:
        ax.axvline(x=result.first_detection, color='orange', linestyle=':',
                   linewidth=1.5,
                   label=f'First detection (sample {result.first_detection})')

    ax.set_title(f'Reconstruction Error: {result.failure_type}')
    ax.set_xlabel('Sample')
    ax.set_ylabel('MSE')
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{result.failure_type}_reconstruction_error.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_data_comparison(result):
    bus_features = ['Bus1_Load', 'Bus1_Avail_Load', 'Bus2_Load', 'Bus2_Avail_Load']
    bus_indices = [MODEL_FEATURES.index(f) for f in bus_features]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, feat, col in zip(axes.flatten(), bus_features, bus_indices):
        ax.plot(result.original_data[:, col], 'b-', alpha=0.5, linewidth=0.5,
                label='Original')
        ax.plot(result.modified_data[:, col], 'r-', alpha=0.5, linewidth=0.5,
                label='Modified')
        ax.axvline(x=result.injection_point, color='green', linestyle='--',
                   alpha=0.8, label='Injection')
        end_point = FAILURE_CONFIGS[result.failure_type].get('end_point')
        if end_point is not None:
            ax.axvline(x=end_point, color='green', linestyle=':', alpha=0.8,
                       label='End')
        ax.set_title(feat)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        ax.legend(fontsize=8)

    fig.suptitle(f'Data Comparison: {result.failure_type}', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{result.failure_type}_data_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_prognostic(result):
    prog = result.prognostic
    if prog is None:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(result.smoothed_errors))

    ax.plot(x, result.smoothed_errors, color='blue', linewidth=1.0,
            label='Smoothed error')
    ax.axhline(y=result.threshold, color='red', linestyle='--', linewidth=1.5,
               label='Threshold')
    ax.axvline(x=result.injection_point, color='green', linestyle='--',
               linewidth=1.5, label='Injection')

    # Regression line in lookback region + extrapolation
    lookback_x = np.arange(prog.lookback_start, prog.lookback_end)
    local_x = lookback_x - prog.lookback_start
    reg_y = prog.slope * local_x + prog.intercept

    ax.plot(lookback_x, reg_y, color='orange', linewidth=2,
            label=f'Regression (R\u00b2={prog.r_squared:.3f})')

    # Extrapolate beyond current data if prediction is in the future
    if (prog.predicted_failure_sample is not None
            and prog.predicted_failure_sample > prog.lookback_end):
        extend_to = min(prog.predicted_failure_sample + 500,
                        prog.lookback_end + 5000)
        extra_x = np.arange(prog.lookback_end, extend_to)
        extra_local = extra_x - prog.lookback_start
        extra_y = prog.slope * extra_local + prog.intercept
        ax.plot(extra_x, extra_y, color='orange', linewidth=2, linestyle=':',
                alpha=0.7)

    if prog.predicted_failure_sample is not None:
        ax.axvline(x=prog.predicted_failure_sample, color='purple',
                   linestyle=':', linewidth=1.5,
                   label=f'Predicted failure (sample {prog.predicted_failure_sample})')

    ax.set_title(f'Prognostic: {result.failure_type}')
    ax.set_xlabel('Sample')
    ax.set_ylabel('MSE')
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{result.failure_type}_prognostic.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_all(results, threshold):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, fault in zip(axes.flatten(), FAULT_TYPES):
        result = results[fault]
        x = np.arange(len(result.smoothed_errors))
        ax.plot(x, result.smoothed_errors, color='blue', linewidth=0.8,
                label='Smoothed')
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.0,
                   label='Threshold')
        ax.axvline(x=result.injection_point, color='green', linestyle='--',
                   linewidth=1.0, label='Injection')
        if result.first_detection is not None:
            ax.axvline(x=result.first_detection, color='orange', linestyle=':',
                       linewidth=1.0,
                       label=f'Detection (+{result.detection_delay})')
        ax.set_title(fault.replace('_', ' ').title())
        ax.set_xlabel('Sample')
        ax.set_ylabel('MSE')
        ax.legend(fontsize=7, loc='upper left')

    fig.suptitle('CBM Evaluation: All Fault Scenarios', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comparison_all_faults.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_spike_analysis(result):
    config = FAILURE_CONFIGS['spikes']
    region_start = max(0, config['injection_point'] - 2000)
    region_end = min(len(result.raw_errors), config['end_point'] + 2000)

    x = np.arange(region_start, region_end)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    axes[0].plot(x, result.raw_errors[region_start:region_end], color='blue',
                 linewidth=0.8, label='Raw error')
    axes[0].axhline(y=result.threshold, color='red', linestyle='--',
                    label='Threshold')
    axes[0].axvline(x=config['injection_point'], color='green', linestyle='--',
                    label='Spike start')
    axes[0].axvline(x=config['end_point'], color='green', linestyle=':',
                    label='Spike end')
    axes[0].set_title('Raw Reconstruction Errors (Spike Region)')
    axes[0].set_ylabel('MSE')
    axes[0].legend(fontsize=8)

    axes[1].plot(x, result.smoothed_errors[region_start:region_end], color='blue',
                 linewidth=0.8, label='Smoothed error (w=50)')
    axes[1].axhline(y=result.threshold, color='red', linestyle='--',
                    label='Threshold')
    axes[1].axvline(x=config['injection_point'], color='green', linestyle='--',
                    label='Spike start')
    axes[1].axvline(x=config['end_point'], color='green', linestyle=':',
                    label='Spike end')
    axes[1].set_title('Smoothed Reconstruction Errors (Spike Region)')
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('MSE')
    axes[1].legend(fontsize=8)

    fig.suptitle('Spike Noise Analysis: Raw vs Smoothed', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'spike_noise_analysis.png',
                dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def write_summary_report(results, threshold, healthy_errors, scale_factors):
    smoothed_healthy = sliding_window_average(healthy_errors, 50)
    sf_str = ', '.join(f'{k}={v}x' for k, v in scale_factors.items())
    lines = [
        'CBM Evaluation Summary Report',
        '=' * 50,
        '',
        f'Threshold:                    {threshold:.6f} (max smoothed healthy * 1.25)',
        f'Injection scale factors:      {sf_str}',
        f'Healthy baseline max (raw):   {np.max(healthy_errors):.6f}',
        f'Healthy baseline max (smooth):{np.max(smoothed_healthy):.6f}',
        f'Healthy baseline mean:        {np.mean(healthy_errors):.6f}',
        f'Healthy baseline std:         {np.std(healthy_errors):.6f}',
        '',
    ]

    for fault_type in FAULT_TYPES:
        result = results[fault_type]
        config = FAILURE_CONFIGS[fault_type]

        lines.append(f'--- {fault_type.replace("_", " ").title()} ---')
        lines.append(f'Description:     {config["description"]}')
        lines.append(f'Injection point: sample {result.injection_point}')
        if 'end_point' in config:
            lines.append(f'End point:       sample {config["end_point"]}')

        if result.first_detection is not None:
            lines.append(f'First detection: sample {result.first_detection}')
            lines.append(f'Detection delay: {result.detection_delay} samples')
        else:
            lines.append('First detection: NOT DETECTED')

        n_anomalies = int(np.sum(result.anomaly_flags))
        lines.append(f'Anomalous windows: {n_anomalies}')

        if result.prognostic is not None:
            p = result.prognostic
            lines.append(f'Prognostic slope:     {p.slope:.8f}')
            lines.append(f'Prognostic R-squared: {p.r_squared:.4f}')
            if p.predicted_failure_sample is not None:
                lines.append(f'Predicted failure:    sample {p.predicted_failure_sample}')
            else:
                lines.append('Predicted failure:    N/A (no positive trend)')

        lines.append('')

    report_path = OUTPUT_DIR / 'summary_report.txt'
    report_path.write_text('\n'.join(lines))
    print(f'Summary report saved to {report_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load model & scaler
    print('Loading model and scaler...')
    model, metadata = load_model(str(MODEL_PATH))
    scaler = joblib.load(str(SCALER_PATH))
    device = next(model.parameters()).device
    print(f'  Model loaded on {device}  (epoch {metadata["epoch"]})')

    # 2. Load healthy data
    print('Loading healthy data...')
    data_dict = joblib.load(str(DATA_PATH))
    healthy_array = joblib_dict_to_array(data_dict)
    print(f'  Data shape: {healthy_array.shape}')

    # 3. Healthy baseline errors & CBM threshold
    #    Threshold is calibrated from *smoothed* healthy errors so it is
    #    comparable to the smoothed faulty errors used for detection.
    print('Computing healthy baseline errors...')
    healthy_errors = compute_reconstruction_errors(
        healthy_array, model, scaler,
        window_size=120, stride=1, batch_size=256, device=device,
    )
    smoothed_healthy = sliding_window_average(healthy_errors, 50)
    threshold = calibrate_threshold(smoothed_healthy, safety_factor=1.25)
    print(f'  CBM Threshold: {threshold:.6f}')

    plot_healthy_baseline(healthy_errors, threshold)
    print('  Saved healthy_baseline.png')

    # 4. Run all fault scenarios
    #    scale_factor amplifies the original injection coefficients so that
    #    the offsets are meaningful relative to bus-load magnitudes (~750 kW std).
    #    slow_drift uses scale=1 (upward drift is strong enough on its own).
    SCALE_FACTORS = {
        'slow_drift': 5,
        'load_imbalance': 10,
        'temporary_reduction': 10,
        'spikes': 10,
    }
    results = {}
    for fault in FAULT_TYPES:
        print(f'\nEvaluating: {fault}...')
        result = run_cbm_evaluation(
            data_dict, fault, model, scaler,
            healthy_errors=healthy_errors,
            threshold=threshold,
            device=device,
            scale_factor=SCALE_FACTORS[fault],
        )
        results[fault] = result

        plot_reconstruction_error(result)
        plot_data_comparison(result)
        if result.prognostic is not None:
            plot_prognostic(result)

        status = (f'detected at +{result.detection_delay} samples'
                  if result.first_detection is not None else 'not detected')
        print(f'  -> {status}')

    # 5. Comparison & spike analysis plots
    plot_comparison_all(results, threshold)
    print('\nSaved comparison_all_faults.png')

    if 'spikes' in results:
        plot_spike_analysis(results['spikes'])
        print('Saved spike_noise_analysis.png')

    # 6. Summary report
    write_summary_report(results, threshold, healthy_errors, SCALE_FACTORS)

    # 7. Save pre-computed results for the Gradio app
    saved = {'threshold': threshold, 'healthy_errors': healthy_errors,
             'scale_factors': SCALE_FACTORS, 'results': {}}
    for fault, r in results.items():
        prog = None
        if r.prognostic:
            p = r.prognostic
            prog = dict(slope=p.slope, intercept=p.intercept,
                        predicted_failure_sample=p.predicted_failure_sample,
                        r_squared=p.r_squared, lookback_start=p.lookback_start,
                        lookback_end=p.lookback_end)
        saved['results'][fault] = dict(
            raw_errors=r.raw_errors, smoothed_errors=r.smoothed_errors,
            anomaly_flags=r.anomaly_flags, injection_point=r.injection_point,
            first_detection=r.first_detection, detection_delay=r.detection_delay,
            prognostic=prog, original_data=r.original_data,
            modified_data=r.modified_data)
    joblib.dump(saved, OUTPUT_DIR / 'results.joblib')
    print(f'Saved results.joblib for Gradio app')

    print(f'\nDone! All outputs in {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
