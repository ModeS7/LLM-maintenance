"""
Gradio UI for Vessel Monitoring System - Matching PowerPoint Design.
"""
import gradio as gr
import numpy as np
from pathlib import Path
from typing import Optional, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data_loader import VesselDataLoader, MODEL_FEATURES
from .inference import AnomalyDetector
from .tools import ToolExecutor
from .llm_agent import create_agent


# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "Data_Pwr_All_S5.txt"
MODEL_PATH = BASE_DIR / "models" / "autoencoder.pt"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
CBM_RESULTS_PATH = BASE_DIR / "docs" / "data" / "results.joblib"

# Global instances
data_loader: Optional[VesselDataLoader] = None
detector: Optional[AnomalyDetector] = None
tool_executor: Optional[ToolExecutor] = None
agent: Optional[Any] = None


def initialize_system():
    """Initialize all system components."""
    global data_loader, detector, tool_executor, agent

    print("Initializing Vessel Monitoring System...")

    print("Loading data...")
    data_loader = VesselDataLoader(str(DATA_PATH), scaler_path=str(SCALER_PATH))
    data_loader.load_data()

    if MODEL_PATH.exists() and SCALER_PATH.exists():
        print("Loading model...")
        detector = AnomalyDetector(str(MODEL_PATH), data_loader)
        tool_executor = ToolExecutor(detector)
        print("Creating LLM agent...")
        agent = create_agent(tool_executor=tool_executor)
    else:
        print("WARNING: Model not found. Run training first.")
        detector = None
        tool_executor = None
        agent = None

    print("System initialized.")


# Load engine image as base64
import base64

def get_engine_image_base64():
    """Load engine SVG and convert to base64."""
    # Try SVG first, then PNG
    for filename in ["diesel_engine.svg", "engine_display.png"]:
        image_path = BASE_DIR / "static" / filename
        if image_path.exists():
            with open(image_path, "rb") as f:
                data = base64.b64encode(f.read()).decode('utf-8')
                mime = "image/svg+xml" if filename.endswith('.svg') else "image/png"
                return data, mime
    return None, None

ENGINE_IMAGE_DATA = None  # Will be loaded on first use


def get_engine_html(time_index: int = None):
    """Generate HTML for engine display with gauges."""
    global ENGINE_IMAGE_DATA

    # Load image on first use
    if ENGINE_IMAGE_DATA is None:
        ENGINE_IMAGE_DATA = get_engine_image_base64()

    if detector:
        if time_index is not None:
            status = detector.get_status_at_index(time_index)
        else:
            status = detector.get_current_status()
        rpm = max(800, int(1200 + status.get('speed', 0) * 50))
        power_mw = status.get('total_power', 0) / 1000

        alerts = []
        if status.get('is_anomaly', False):
            for var, error in status.get('top_contributors', [])[:3]:
                alerts.append(f"detected failure: {var}")
    else:
        rpm = 1500
        power_mw = 5.0
        alerts = []

    alerts_html = ""
    for alert in alerts:
        alerts_html += f'<div style="background:#fef2f2; border-left:4px solid #ef4444; padding:8px 15px; border-radius:5px; font-size:13px; color:#dc2626; white-space: nowrap;">{alert}</div>'

    if not alerts_html:
        alerts_html = '<div style="background:#f0fdf4; border-left:4px solid #22c55e; padding:8px 15px; border-radius:5px; color:#16a34a; font-size:14px; font-weight: 500;">All systems normal</div>'

    # Use base64 image or fallback
    img_data, img_mime = ENGINE_IMAGE_DATA if ENGINE_IMAGE_DATA else (None, None)
    if img_data:
        img_src = f"data:{img_mime};base64,{img_data}"
    else:
        img_src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='380' height='220'%3E%3Crect fill='%23f3f4f6' width='380' height='220' rx='10'/%3E%3Ctext x='190' y='110' fill='%236b7280' text-anchor='middle' font-size='20'%3EMarine Diesel Engine%3C/text%3E%3C/svg%3E"

    return f'''
    <div style="padding: 25px 30px; border-radius: 12px;">
        <div style="display: flex; justify-content: center; align-items: center; gap: 30px;">

            <!-- Left Gauge - RPM -->
            <div style="display: flex; flex-direction: column; align-items: center; flex-shrink: 0;">
                <div style="background: linear-gradient(135deg, #3b82f6, #6366f1); border-radius: 50%; width: 130px; height: 130px; display: flex; flex-direction: column; align-items: center; justify-content: center; font-weight: bold; box-shadow: 0 6px 25px rgba(0,0,0,0.2); border: 4px solid #3b82f6;">
                    <span style="font-size: 32px; color: #22c55e;">{rpm}</span>
                    <span style="font-size: 14px; color: white;">RPM</span>
                </div>
            </div>

            <!-- Center - Engine Image -->
            <div style="flex: 1; display: flex; flex-direction: column; align-items: center; max-width: 700px;">
                <img src="{img_src}"
                     style="width: 100%; height: auto; border-radius: 12px; box-shadow: 0 8px 30px rgba(0,0,0,0.2);"
                     alt="Marine Diesel Engine">
                <!-- Alerts below image -->
                <div style="margin-top: 15px; display: flex; gap: 10px; flex-wrap: wrap; justify-content: center;">
                    {alerts_html}
                </div>
            </div>

            <!-- Right Gauge - Power -->
            <div style="display: flex; flex-direction: column; align-items: center; flex-shrink: 0;">
                <div style="background: linear-gradient(135deg, #3b82f6, #6366f1); border-radius: 50%; width: 130px; height: 130px; display: flex; flex-direction: column; align-items: center; justify-content: center; font-weight: bold; box-shadow: 0 6px 25px rgba(0,0,0,0.2); border: 4px solid #3b82f6;">
                    <span style="font-size: 32px; color: #22c55e;">{power_mw:.1f}</span>
                    <span style="font-size: 14px; color: white;">MW</span>
                </div>
            </div>
        </div>
    </div>
    '''


def get_data_button_labels(time_index: int = None):
    """Return current values for data buttons."""
    if detector:
        if time_index is not None:
            status = detector.get_status_at_index(time_index)
        else:
            status = detector.get_current_status()
        return (
            f"Bus1 Load\n{status.get('bus1_load', 0):.0f} kW",
            f"Bus2 Load\n{status.get('bus2_load', 0):.0f} kW",
            f"Speed\n{status.get('speed', 0):.1f} kts",
            f"Position\n{status.get('latitude', 0):.2f}°N"
        )
    return ("Bus1 Load\n-- kW", "Bus2 Load\n-- kW", "Speed\n-- kts", "Position\n--°N")


def get_variables_html(time_index: int = None):
    """Generate variable display boxes."""
    if detector:
        if time_index is not None:
            status = detector.get_status_at_index(time_index)
        else:
            status = detector.get_current_status()
        vars_data = [
            ("Bus1 Load", f"{status.get('bus1_load', 0):.0f} kW"),
            ("Bus2 Load", f"{status.get('bus2_load', 0):.0f} kW"),
            ("Speed", f"{status.get('speed', 0):.1f} kts"),
            ("Position", f"{status.get('latitude', 0):.2f}°N"),
        ]
    else:
        vars_data = [
            ("Bus1 Load", "-- kW"),
            ("Bus2 Load", "-- kW"),
            ("Speed", "-- kts"),
            ("Position", "--°N"),
        ]

    boxes_html = ""
    for name, value in vars_data:
        boxes_html += f'''
        <div style="text-align: center; margin: 0 10px; flex: 1; min-width: 140px;">
            <div style="border: 1px solid rgba(0,0,0,0.1); border-radius: 12px; padding: 15px 20px; margin-bottom: 8px;">
                <div style="font-weight: 600; font-size: 18px; color: #22c55e;">{value}</div>
            </div>
            <div style="font-size: 13px; opacity: 0.7;">{name}</div>
        </div>
        '''

    return f'<div style="display: flex; justify-content: center; margin-top: 20px; flex-wrap: wrap; gap: 10px;">{boxes_html}</div>'


def get_realtime_page_html(time_index: int = None):
    """Generate complete real-time page HTML (engine display only, buttons are Gradio components)."""
    return f'''
    <div style="padding: 25px; border-radius: 12px;">
        {get_engine_html(time_index)}
    </div>
    '''


def create_anomaly_chart():
    """Create chart with anomaly markers based on reconstruction error."""
    if detector is None or data_loader is None:
        return go.Figure()

    try:
        # Get reconstruction comparison to find actual anomalies
        recon_data = detector.get_reconstruction_comparison('Bus1_Load', hours=1)
        if 'actual' not in recon_data:
            return go.Figure()

        actual = recon_data['actual']
        reconstructed = recon_data['reconstructed']
        timestamps = list(range(len(actual)))

        # Calculate reconstruction error at each point
        errors = [abs(a - r) for a, r in zip(actual, reconstructed)]

        # Find anomaly threshold (e.g., points with error > 95th percentile)
        error_threshold = np.percentile(errors, 95)

        # Find anomaly indices
        anomaly_indices = [i for i, e in enumerate(errors) if e > error_threshold]
        anomaly_values = [actual[i] for i in anomaly_indices]

        fig = go.Figure()

        # Main time series
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=actual,
            mode='lines',
            name='Bus1_Load',
            line=dict(color='#1E90FF', width=2)
        ))

        # Mark actual anomalies (high reconstruction error)
        if anomaly_indices:
            fig.add_trace(go.Scatter(
                x=anomaly_indices,
                y=anomaly_values,
                mode='markers',
                name='Anomaly',
                marker=dict(color='red', size=8, symbol='circle'),
                hovertemplate='Anomaly<br>Time: %{x}<br>Value: %{y:.1f}<br>High reconstruction error<extra></extra>'
            ))

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Value (kW)",
            plot_bgcolor='white',
            paper_bgcolor='rgba(135,206,235,0.3)',
            height=280,
            margin=dict(l=50, r=30, t=30, b=50),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )

        return fig
    except Exception:
        return go.Figure()


def create_comparison_chart():
    """Create actual vs predicted comparison chart."""
    if detector is None or data_loader is None:
        return go.Figure()

    try:
        recon_data = detector.get_reconstruction_comparison('Bus1_Load', hours=1)
        # Check for error message (string), not error values (list)
        if 'actual' not in recon_data:
            return go.Figure()

        actual = recon_data['actual']
        reconstructed = recon_data['reconstructed']
        x = list(range(len(actual)))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x, y=actual,
            mode='lines',
            name='Actual',
            line=dict(color='#1E90FF', width=1.5)
        ))

        fig.add_trace(go.Scatter(
            x=x, y=reconstructed,
            mode='lines',
            name='Predicted',
            line=dict(color='#FF6B6B', width=1.5)
        ))

        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='rgba(135,206,235,0.3)',
            height=280,
            margin=dict(l=50, r=30, t=30, b=50),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )

        return fig
    except Exception:
        return go.Figure()


def apply_chart_styling(fig: go.Figure) -> go.Figure:
    """Apply clean styling to a plotly figure that works in light and dark modes."""
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent - inherits from container
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent
        font=dict(color='#374151'),  # Neutral gray that works in both modes
        xaxis=dict(
            gridcolor='rgba(156, 163, 175, 0.3)',
            linecolor='rgba(156, 163, 175, 0.5)',
        ),
        yaxis=dict(
            gridcolor='rgba(156, 163, 175, 0.3)',
            linecolor='rgba(156, 163, 175, 0.5)',
        ),
    )
    return fig


def create_variable_chart(variable: str = 'Bus1_Load', time_index: int = None):
    """Create chart for a selected variable with anomaly markers.

    Args:
        variable: Variable name to chart
        time_index: If provided, center chart around this index. Otherwise use latest data.
    """
    if detector is None or data_loader is None:
        return go.Figure()

    try:
        if time_index is not None:
            recon_data = detector.get_reconstruction_at_index(variable, time_index, hours=1)
        else:
            recon_data = detector.get_reconstruction_comparison(variable, hours=1)
        if 'actual' not in recon_data:
            return go.Figure()

        actual = recon_data['actual']
        reconstructed = recon_data['reconstructed']
        errors = [abs(a - r) for a, r in zip(actual, reconstructed)]
        x = list(range(len(actual)))

        # Find anomalies (error > 95th percentile)
        error_threshold = np.percentile(errors, 95)
        anomaly_indices = [i for i, e in enumerate(errors) if e > error_threshold]
        anomaly_values = [actual[i] for i in anomaly_indices]

        fig = go.Figure()

        # Actual values
        fig.add_trace(go.Scatter(
            x=x,
            y=actual,
            mode='lines',
            name='Actual',
            line=dict(color='#4a9eff', width=2)
        ))

        # Reconstructed values
        fig.add_trace(go.Scatter(
            x=x,
            y=reconstructed,
            mode='lines',
            name='Reconstructed',
            line=dict(color='#4ade80', width=2)
        ))

        # Anomaly markers
        if anomaly_indices:
            fig.add_trace(go.Scatter(
                x=anomaly_indices,
                y=anomaly_values,
                mode='markers',
                name='Anomaly',
                marker=dict(color='#f87171', size=10, symbol='circle'),
                hovertemplate='Anomaly<br>Time: %{x}<br>Value: %{y:.2f}<extra></extra>'
            ))

        fig.update_layout(
            title=f'{variable} - Actual vs Reconstructed',
            xaxis_title='Time Index',
            yaxis_title='Value',
            height=400,
            margin=dict(l=60, r=30, t=50, b=50),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )

        return apply_chart_styling(fig)
    except Exception:
        return go.Figure()


def create_total_error_chart(time_index: int = None):
    """Create chart showing total reconstruction error over time.

    Args:
        time_index: If provided, center chart around this index. Otherwise use latest data.
    """
    if detector is None or data_loader is None:
        return go.Figure()

    try:
        if time_index is not None:
            recon_data = detector.get_all_features_reconstruction_at_index(time_index, hours=1)
        else:
            recon_data = detector.get_all_features_reconstruction(hours=1)
        if 'total_error' not in recon_data:
            return go.Figure()

        total_error = recon_data['total_error']
        x = list(range(len(total_error)))

        # Calculate threshold (95th percentile)
        threshold = np.percentile(total_error, 95)

        fig = go.Figure()

        # Total error line
        fig.add_trace(go.Scatter(
            x=x,
            y=total_error,
            mode='lines',
            name='Total Error',
            line=dict(color='#4a9eff', width=2),
            fill='tozeroy',
            fillcolor='rgba(74, 158, 255, 0.2)'
        ))

        # Threshold line
        fig.add_hline(
            y=threshold,
            line_dash='dash',
            line_color='#fb923c',
            annotation_text='95th Percentile Threshold',
            annotation_position='top right',
            annotation_font_color='#fb923c'
        )

        fig.update_layout(
            title='Total Reconstruction Error (All Features)',
            xaxis_title='Time Index',
            yaxis_title='Total Error',
            height=400,
            margin=dict(l=60, r=30, t=50, b=50),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )

        return apply_chart_styling(fig)
    except Exception:
        return go.Figure()


def create_threshold_heatmap():
    """Create heatmap showing per-feature errors relative to threshold."""
    if detector is None or data_loader is None:
        return go.Figure()

    try:
        recon_data = detector.get_all_features_reconstruction(hours=1)
        if 'errors_normalized' not in recon_data:
            return go.Figure()

        # Use normalized errors (same scale as threshold)
        errors = recon_data['errors_normalized']  # (n_timesteps, 16)
        feature_names = recon_data['feature_names']
        threshold = detector.threshold

        # Transpose for heatmap (features on y-axis, time on x-axis)
        errors_transposed = errors.T  # (16, n_timesteps)

        # Scale errors relative to threshold (0 = no error, 1 = at threshold, 2 = 200%)
        errors_scaled = errors_transposed / threshold

        # Custom colorscale: green up to 80%, then yellow to red up to 200%
        # Values are scaled where 0.8 = 80% of threshold, 2.0 = 200% of threshold
        colorscale = [
            [0.0, '#22c55e'],    # 0% - bright green
            [0.4, '#22c55e'],    # 80% (0.8/2.0) - still green (healthy)
            [0.5, '#eab308'],    # 100% (1.0/2.0) - yellow (at threshold)
            [0.75, '#f97316'],   # 150% - orange (warning)
            [1.0, '#dc2626'],    # 200%+ - red (critical)
        ]

        fig = go.Figure()

        fig.add_trace(go.Heatmap(
            z=errors_scaled,
            y=feature_names,
            colorscale=colorscale,
            zmin=0,
            zmax=2.0,  # Cap at 200% of threshold
            colorbar=dict(
                title=dict(text='% of Threshold'),
                tickvals=[0, 0.8, 1.0, 1.5, 2.0],
                ticktext=['0%', '80%', '100%', '150%', '200%+']
            ),
            hovertemplate='Feature: %{y}<br>Time: %{x}<br>Error: %{z:.0%} of threshold<extra></extra>'
        ))

        fig.update_layout(
            title=f'Feature Health Heatmap (threshold={threshold:.4f})',
            xaxis_title='Time Index',
            yaxis_title='Feature',
            height=500,
            margin=dict(l=150, r=30, t=50, b=50),
        )

        return apply_chart_styling(fig)
    except Exception:
        return go.Figure()


def chat_respond(message, history):
    """Process chat message and return response."""
    if not message.strip():
        return history, ""

    if agent is None:
        response = "System not ready. Please ensure the model is trained."
    else:
        try:
            response = agent.chat(message)
        except Exception as e:
            response = f"Error: {str(e)}"

    # Gradio 6 expects messages as dicts with 'role' and 'content'
    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history, ""


# CSS Styling - Works in both light and dark modes
CUSTOM_CSS = """
/* ========== LIGHT MODE (default) ========== */
:root {
    --accent-blue: #3b82f6;
    --accent-green: #22c55e;
    --accent-orange: #f59e0b;
    --accent-red: #ef4444;
    --accent-purple: #8b5cf6;
}

/* ========== HOME PAGE ========== */
.home-btn {
    background: linear-gradient(135deg, var(--accent-purple) 0%, #7c3aed 100%) !important;
    color: white !important;
    font-size: 20px !important;
    font-weight: 600 !important;
    padding: 25px 50px !important;
    border-radius: 12px !important;
    border: none !important;
    min-width: 400px !important;
    box-shadow: 0 4px 20px rgba(139, 92, 246, 0.3) !important;
    transition: all 0.3s ease !important;
}
.home-btn:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 6px 25px rgba(139, 92, 246, 0.4) !important;
}

/* ========== NAVIGATION ========== */
.back-btn {
    font-size: 24px !important;
    padding: 8px 16px !important;
    border-radius: 8px !important;
}

.interaction-btn {
    background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%) !important;
    color: #1a1a1a !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    border-radius: 8px !important;
}

/* ========== CHAT STYLES ========== */
.message {
    border-radius: 12px !important;
    padding: 12px 16px !important;
}

/* ========== STATUS INDICATORS ========== */
.status-healthy { color: var(--accent-green) !important; }
.status-warning { color: var(--accent-orange) !important; }
.status-critical { color: var(--accent-red) !important; }

/* ========== DYNAMIC HEIGHT CHAT ========== */
.chat-fullpage {
    height: calc(100vh - 80px) !important;
    /* Let Gradio handle flex layout for history sidebar */
}

/* ========== RESPONSIVE ========== */
@media (max-width: 768px) {
    .home-btn {
        min-width: 280px !important;
        font-size: 18px !important;
        padding: 20px 30px !important;
    }
}
"""


# ---------------------------------------------------------------------------
# CBM Evaluation helpers
# ---------------------------------------------------------------------------
_cbm_cache: dict = {}          # 'precomputed' -> saved dict
_cbm_live_cache: dict = {}     # fault_type -> result dict (from live runs)


def _load_cbm_precomputed():
    """Load pre-computed CBM results (cached after first call)."""
    if 'precomputed' not in _cbm_cache:
        import joblib as _jl
        if CBM_RESULTS_PATH.exists():
            _cbm_cache['precomputed'] = _jl.load(CBM_RESULTS_PATH)
        else:
            _cbm_cache['precomputed'] = None
    return _cbm_cache['precomputed']


def _get_fault_data(fault_type):
    """Return the result dict for *fault_type* (precomputed or live-cached)."""
    if fault_type in _cbm_live_cache:
        return _cbm_live_cache[fault_type]
    saved = _load_cbm_precomputed()
    if saved and fault_type in saved['results']:
        return saved['results'][fault_type]
    return None


def _get_healthy_errors():
    """Return healthy raw errors array from precomputed data."""
    saved = _load_cbm_precomputed()
    return saved['healthy_errors'] if saved else None


def _no_data_fig(msg="No data"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False, font=dict(size=16))
    fig.update_layout(height=400)
    return fig


def cbm_live_compute(fault_type, scale_factor, injection_point=None):
    """Full GPU re-computation for one fault. Result is cached."""
    import joblib as _jl
    from .cbm import (joblib_dict_to_array, compute_reconstruction_errors,
                      run_cbm_evaluation, sliding_window_average,
                      calibrate_threshold)
    from .model import load_model

    model, _ = load_model(str(MODEL_PATH))
    scaler = _jl.load(str(SCALER_PATH))
    device = next(model.parameters()).device
    data_path = BASE_DIR / "docs" / "data" / "variable_of_interest_for_PCC.joblib"
    data_dict = _jl.load(str(data_path))

    healthy_arr = joblib_dict_to_array(data_dict)
    healthy_err = compute_reconstruction_errors(healthy_arr, model, scaler,
                                                device=device)
    # Store healthy errors for slider-based reprocessing
    _cbm_cache.setdefault('precomputed', {})['healthy_errors'] = healthy_err

    smoothed_h = sliding_window_average(healthy_err, 50)
    threshold = calibrate_threshold(smoothed_h, 1.20)

    r = run_cbm_evaluation(data_dict, fault_type, model, scaler,
                           healthy_errors=healthy_err, threshold=threshold,
                           device=device, scale_factor=scale_factor,
                           injection_point=injection_point)
    result_dict = dict(
        raw_errors=r.raw_errors, smoothed_errors=r.smoothed_errors,
        anomaly_flags=r.anomaly_flags, injection_point=r.injection_point,
        first_detection=r.first_detection, detection_delay=r.detection_delay,
        original_data=r.original_data, modified_data=r.modified_data,
        prognostic=None,
    )
    _cbm_live_cache[fault_type] = result_dict
    return result_dict


# ---- chart builders -------------------------------------------------------

def _build_error_chart(fault_type, smoothing_window, safety_factor):
    """Reconstruction-error chart, reprocessed from raw errors."""
    from .cbm import sliding_window_average, FAILURE_CONFIGS

    data = _get_fault_data(fault_type)
    healthy = _get_healthy_errors()
    if data is None or healthy is None:
        return _no_data_fig("Run `python run_cbm_evaluation.py` first."), ""

    raw = data['raw_errors']
    smoothed = sliding_window_average(raw, smoothing_window)
    healthy_sm = sliding_window_average(healthy, smoothing_window)
    threshold = float(np.max(healthy_sm) * safety_factor)
    flags = smoothed > threshold

    inj = data['injection_point']
    anomaly_idx = np.where(flags)[0]
    post = anomaly_idx[anomaly_idx >= inj] if len(anomaly_idx) else np.array([])
    first_det = int(post[0]) if len(post) else None
    delay = (first_det - inj) if first_det is not None else None

    x = np.arange(len(raw))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=raw, mode='lines', name='Raw error',
                             line=dict(color='rgba(100,160,255,0.25)', width=1)))
    fig.add_trace(go.Scatter(x=x, y=smoothed, mode='lines',
                             name=f'Smoothed (w={smoothing_window})',
                             line=dict(color='#3b82f6', width=2)))
    fig.add_hline(y=threshold, line_dash='dash', line_color='#ef4444',
                  annotation_text=f'Threshold ({threshold:.4f})',
                  annotation_position='top left',
                  annotation_font_color='#ef4444')
    fig.add_vline(x=inj, line_dash='dash', line_color='#22c55e',
                  annotation_text=f'Injection ({inj})',
                  annotation_position='top right',
                  annotation_font_color='#22c55e')

    # Anomaly shading
    if np.any(flags):
        diff = np.diff(flags.astype(int))
        starts = (np.where(diff == 1)[0] + 1).tolist()
        ends = (np.where(diff == -1)[0] + 1).tolist()
        if flags[0]:
            starts = [0] + starts
        if flags[-1]:
            ends = ends + [len(flags)]
        for s, e in zip(starts, ends):
            fig.add_vrect(x0=s, x1=e, fillcolor='rgba(239,68,68,0.12)',
                          line_width=0)

    if first_det is not None:
        fig.add_vline(x=first_det, line_dash='dot', line_color='#f59e0b',
                      annotation_text=f'Detection (+{delay})',
                      annotation_position='top left',
                      annotation_font_color='#f59e0b')

    title = fault_type.replace('_', ' ').title()
    fig.update_layout(title=f'Reconstruction Error \u2014 {title}',
                      xaxis_title='Sample', yaxis_title='MSE',
                      height=500, hovermode='x unified',
                      margin=dict(l=60, r=30, t=60, b=50),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                  xanchor='center', x=0.5))

    det_str = f"sample {first_det} (delay: {delay})" if first_det else "NOT DETECTED"
    n_anom = int(np.sum(flags))
    summary = (f"**{title}**  |  Injection: sample {inj}  |  "
               f"First detection: {det_str}  |  "
               f"Anomalous windows: {n_anom}  |  "
               f"Threshold: {threshold:.4f}")
    return apply_chart_styling(fig), summary


def _build_data_comparison(fault_type):
    """4-subplot chart: original vs modified bus loads."""
    data = _get_fault_data(fault_type)
    if data is None:
        return _no_data_fig()

    bus = ['Bus1_Load', 'Bus1_Avail_Load', 'Bus2_Load', 'Bus2_Avail_Load']
    bus_idx = [MODEL_FEATURES.index(f) for f in bus]
    inj = data['injection_point']
    orig = data['original_data']
    mod = data['modified_data']
    x = np.arange(len(orig))

    fig = make_subplots(rows=2, cols=2, subplot_titles=bus,
                        horizontal_spacing=0.08, vertical_spacing=0.12)
    for i, (feat, ci) in enumerate(zip(bus, bus_idx)):
        r, c = i // 2 + 1, i % 2 + 1
        fig.add_trace(go.Scatter(x=x, y=orig[:, ci], mode='lines',
                                 name='Original', legendgroup='orig',
                                 showlegend=(i == 0),
                                 line=dict(color='#3b82f6', width=1)),
                      row=r, col=c)
        fig.add_trace(go.Scatter(x=x, y=mod[:, ci], mode='lines',
                                 name='Modified', legendgroup='mod',
                                 showlegend=(i == 0),
                                 line=dict(color='#ef4444', width=1)),
                      row=r, col=c)
        fig.add_vline(x=inj, line_dash='dash', line_color='#22c55e',
                      line_width=1, row=r, col=c)

    title = fault_type.replace('_', ' ').title()
    fig.update_layout(title=f'Data Comparison \u2014 {title}',
                      height=600, hovermode='x unified',
                      margin=dict(l=60, r=30, t=80, b=50),
                      legend=dict(orientation='h', yanchor='bottom', y=1.04,
                                  xanchor='center', x=0.5))
    return apply_chart_styling(fig)


def _build_prognostic(fault_type, smoothing_window, safety_factor):
    """Linear-regression prognostic chart (slow_drift / load_imbalance)."""
    from .cbm import sliding_window_average, estimate_time_to_failure

    if fault_type not in ('slow_drift', 'load_imbalance'):
        fig = _no_data_fig("Prognostic available for Slow Drift and Load Imbalance only.")
        return fig, ""

    data = _get_fault_data(fault_type)
    healthy = _get_healthy_errors()
    if data is None or healthy is None:
        return _no_data_fig(), ""

    raw = data['raw_errors']
    smoothed = sliding_window_average(raw, smoothing_window)
    healthy_sm = sliding_window_average(healthy, smoothing_window)
    threshold = float(np.max(healthy_sm) * safety_factor)
    inj = data['injection_point']

    prog = estimate_time_to_failure(smoothed, threshold, lookback=2000)

    x = np.arange(len(smoothed))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=smoothed, mode='lines',
                             name='Smoothed error',
                             line=dict(color='#3b82f6', width=2)))
    fig.add_hline(y=threshold, line_dash='dash', line_color='#ef4444',
                  annotation_text='Threshold',
                  annotation_font_color='#ef4444')
    fig.add_vline(x=inj, line_dash='dash', line_color='#22c55e',
                  annotation_text='Injection', annotation_font_color='#22c55e')

    # Regression line
    lb_x = np.arange(prog.lookback_start, prog.lookback_end)
    loc_x = lb_x - prog.lookback_start
    reg_y = prog.slope * loc_x + prog.intercept
    fig.add_trace(go.Scatter(x=lb_x, y=reg_y, mode='lines',
                             name=f'Regression (R\u00b2={prog.r_squared:.3f})',
                             line=dict(color='#f59e0b', width=3)))

    # Extrapolation
    if (prog.predicted_failure_sample is not None
            and prog.predicted_failure_sample > prog.lookback_end):
        ext_x = np.arange(prog.lookback_end,
                          min(prog.predicted_failure_sample + 500,
                              prog.lookback_end + 5000))
        ext_loc = ext_x - prog.lookback_start
        ext_y = prog.slope * ext_loc + prog.intercept
        fig.add_trace(go.Scatter(x=ext_x, y=ext_y, mode='lines',
                                 name='Extrapolation',
                                 line=dict(color='#f59e0b', width=2,
                                           dash='dot')))

    if prog.predicted_failure_sample is not None:
        fig.add_vline(x=prog.predicted_failure_sample, line_dash='dot',
                      line_color='#8b5cf6',
                      annotation_text=f'Predicted failure ({prog.predicted_failure_sample})',
                      annotation_font_color='#8b5cf6')

    title = fault_type.replace('_', ' ').title()
    fig.update_layout(title=f'Prognostic \u2014 {title}',
                      xaxis_title='Sample', yaxis_title='MSE',
                      height=480, hovermode='x unified',
                      margin=dict(l=60, r=30, t=60, b=50),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                  xanchor='center', x=0.5))

    pred = prog.predicted_failure_sample
    info = (f"R\u00b2 = {prog.r_squared:.4f}  |  "
            f"Predicted failure: sample {pred}" if pred else
            f"R\u00b2 = {prog.r_squared:.4f}  |  No positive trend")
    return apply_chart_styling(fig), info


def create_app() -> gr.Blocks:
    """Create the Gradio application."""

    with gr.Blocks(title="Vessel Monitoring System") as app:

        # Global state for selected time index (shared across all pages)
        if detector:
            test_info = detector.get_test_data_info()
            initial_index = test_info['end_index']
        else:
            initial_index = 0
        selected_time_state = gr.State(value=initial_index)

        # ============== HOME PAGE ==============
        with gr.Column(visible=True) as home_page:
            gr.HTML('''
                <div style="padding: 60px 40px; border-radius: 16px; min-height: 480px;">
                    <h1 style="text-align: center; margin-bottom: 15px; font-size: 36px; font-weight: 600;">
                        Vessel Monitoring System
                    </h1>
                    <p style="text-align: center; opacity: 0.7; margin-bottom: 50px; font-size: 16px;">
                        AI-powered monitoring for offshore vessel operations
                    </p>
                    <div style="display: flex; flex-direction: column; align-items: center; gap: 20px;">
            ''')
            btn_realtime = gr.Button("REAL TIME MONITORING", elem_classes=["home-btn"])
            btn_chats = gr.Button("AI CHAT ASSISTANT", elem_classes=["home-btn"])
            btn_cbm = gr.Button("CBM EVALUATION", elem_classes=["home-btn"])
            gr.HTML('''
                    </div>
                    <p style="text-align: center; opacity: 0.5; margin-top: 40px; font-size: 13px;">
                        M/S Olympic Hera • Offshore Construction Vessel
                    </p>
                </div>
            ''')

        # ============== REAL TIME PAGE ==============
        with gr.Column(visible=False) as realtime_page:
            with gr.Row():
                back_btn_rt = gr.Button("Back", elem_classes=["back-btn"], scale=0, min_width=100)
                gr.HTML('<div style="flex:1;"><h2 style="text-align:center; margin:0;">Real Time Monitoring</h2></div>')
                view_charts_btn = gr.Button("Charts", variant="secondary")

            # Get test data range for slider
            if detector:
                test_info = detector.get_test_data_info()
                slider_min = test_info['start_index']
                slider_max = test_info['end_index']
                slider_value = slider_max  # Start at latest
                initial_time_str = test_info['end_time']
            else:
                slider_min, slider_max, slider_value = 0, 100, 100
                initial_time_str = "No data"

            realtime_display = gr.HTML(value=get_realtime_page_html)

            # Data buttons row (Gradio buttons with live values)
            initial_labels = get_data_button_labels()
            with gr.Row():
                btn_bus1 = gr.Button(initial_labels[0])
                btn_bus2 = gr.Button(initial_labels[1])
                btn_speed = gr.Button(initial_labels[2])
                btn_position = gr.Button(initial_labels[3])

            # Time slider for navigating test data
            gr.Markdown("**Navigate through test data:**")
            with gr.Row():
                time_slider = gr.Slider(
                    minimum=slider_min,
                    maximum=slider_max,
                    value=slider_value,
                    step=720,  # 1 hour steps (720 samples × 5 seconds)
                    label="Timeline",
                    elem_id="time-slider"
                )
                time_display = gr.Textbox(
                    value=initial_time_str,
                    label="Selected Time",
                    interactive=False,
                    scale=0,
                    min_width=200
                )

            refresh_rt = gr.Button("Refresh Data", variant="primary")

        # ============== CHARTS PAGE ==============
        with gr.Column(visible=False) as charts_page:
            with gr.Row():
                back_btn_charts = gr.Button("Back", elem_classes=["back-btn"], scale=0, min_width=100)
                gr.HTML('<div style="flex:1;"><h2 style="text-align:center; margin:0;">Analytics & Charts</h2></div>')
                interaction_btn = gr.Button("Live View", elem_classes=["interaction-btn"])

            with gr.Tabs():
                with gr.TabItem("Variable Explorer"):
                    variable_dropdown = gr.Dropdown(
                        choices=MODEL_FEATURES,
                        value='Bus1_Load',
                        label="Select Variable",
                        interactive=True
                    )
                    variable_chart = gr.Plot(value=lambda: create_variable_chart('Bus1_Load'))

                with gr.TabItem("Total Error"):
                    total_error_chart = gr.Plot(value=create_total_error_chart)

            # Wire up dropdown change event (uses selected_time_state defined at app level)
            def on_variable_change(variable, time_index):
                """Update chart when variable changes, using current time index."""
                return create_variable_chart(variable, time_index=time_index)

            variable_dropdown.change(
                fn=on_variable_change,
                inputs=[variable_dropdown, selected_time_state],
                outputs=[variable_chart]
            )

        # ============== CHATS PAGE ==============
        with gr.Column(visible=False, elem_classes=["chat-fullpage"]) as chats_page:
            with gr.Row():
                back_btn_chat = gr.Button("Back", elem_classes=["back-btn"], scale=0, min_width=100)
                gr.HTML('<div style="flex:1;"><h2 style="text-align:center; margin:0;">AI Assistant</h2></div>')
                gr.HTML('<div style="width:100px;"></div>')

            # Simple chat function for ChatInterface
            def chat_fn(message, history):
                """Process chat message."""
                if not message.strip():
                    return ""
                if agent is None:
                    return "System not ready. Please ensure the model is trained."
                try:
                    return agent.chat(message)
                except Exception as e:
                    return f"Error: {str(e)}"

            # Quick prompt buttons
            gr.Markdown("**Quick prompts:**", elem_classes=["quick-prompts-label"])
            with gr.Row():
                btn_status = gr.Button("Vessel status", size="sm", scale=1)
                btn_electrical = gr.Button("Electrical readings", size="sm", scale=1)
                btn_anomalies = gr.Button("Anomalies detected?", size="sm", scale=1)
                btn_propulsion = gr.Button("Propulsion power", size="sm", scale=1)

            # ChatInterface wrapped in container for JS targeting
            with gr.Column(elem_id="vessel-chat"):
                chat = gr.ChatInterface(
                    fn=chat_fn,
                    title=None,
                    description="Ask me about vessel status, power systems, anomalies, and more.",
                    save_history=True,
                    fill_height=True,  # Re-enabled: bug fixed in Gradio 6.x (PR #10372)
                )

            # JS to click submit button after textbox is populated
            submit_js = "() => document.querySelector('#vessel-chat button.primary')?.click()"

            # Connect buttons to populate textbox and auto-submit
            btn_status.click(
                fn=lambda: "What is the current vessel status?",
                outputs=chat.textbox
            ).then(fn=None, js=submit_js)
            btn_electrical.click(
                fn=lambda: "Show me the electrical system readings",
                outputs=chat.textbox
            ).then(fn=None, js=submit_js)
            btn_anomalies.click(
                fn=lambda: "Are there any anomalies detected?",
                outputs=chat.textbox
            ).then(fn=None, js=submit_js)
            btn_propulsion.click(
                fn=lambda: "What is the propulsion power output?",
                outputs=chat.textbox
            ).then(fn=None, js=submit_js)

        # ============== CBM EVALUATION PAGE ==============
        with gr.Column(visible=False) as cbm_page:
            with gr.Row():
                back_btn_cbm = gr.Button("Back", elem_classes=["back-btn"],
                                         scale=0, min_width=100)
                gr.HTML('<div style="flex:1;"><h2 style="text-align:center; margin:0;">'
                        'CBM Failure Injection Evaluation</h2></div>')

            with gr.Row():
                cbm_fault = gr.Radio(
                    choices=["slow_drift", "load_imbalance",
                             "temporary_reduction", "spikes"],
                    value="slow_drift", label="Fault Scenario")
            with gr.Row():
                cbm_sw = gr.Slider(minimum=5, maximum=200, value=50, step=5,
                                   label="Smoothing window")
                cbm_sf = gr.Slider(minimum=1.0, maximum=2.0, value=1.20,
                                   step=0.05, label="Safety factor")
            with gr.Row():
                cbm_scale = gr.Slider(minimum=1, maximum=30, value=10, step=1,
                                      label="Injection scale factor (live compute only)")
                cbm_inj = gr.Slider(minimum=1000, maximum=25000, value=20000,
                                    step=500,
                                    label="Injection point — sample (live compute only)")
                cbm_live_btn = gr.Button("Live Compute (GPU)", variant="secondary")

            cbm_summary = gr.Markdown("Select a fault scenario above.")

            with gr.Tabs(elem_id="cbm-tabs") as cbm_tabs:
                with gr.TabItem("Reconstruction Error"):
                    cbm_error_chart = gr.Plot()
                with gr.TabItem("Data Comparison"):
                    cbm_data_chart = gr.Plot()
                with gr.TabItem("Prognostic"):
                    cbm_prog_info = gr.Markdown()
                    cbm_prog_chart = gr.Plot()

            # Resize Plotly charts when switching tabs (they render
            # with wrong dimensions while their tab is hidden)
            cbm_tabs.change(fn=None, js="""
                () => setTimeout(() => {
                    document.querySelectorAll('#cbm-tabs .js-plotly-plot')
                        .forEach(p => Plotly.Plots.resize(p));
                }, 50)
            """)

            # --- instant updates (from pre-computed / cached raw errors) ----
            _cbm_outputs = [cbm_error_chart, cbm_summary,
                            cbm_data_chart, cbm_prog_chart, cbm_prog_info]
            _cbm_inputs = [cbm_fault, cbm_sw, cbm_sf]
            _cbm_empty = [go.Figure(), "", go.Figure(), go.Figure(), ""]

            def _refresh_all(fault, sw, sf):
                # Guard: Gradio may fire events with None when hidden
                # components first render
                if not fault or sw is None or sf is None:
                    return _cbm_empty
                err_fig, summ = _build_error_chart(fault, int(sw), sf)
                data_fig = _build_data_comparison(fault)
                prog_fig, prog_info = _build_prognostic(fault, int(sw), sf)
                return err_fig, summ, data_fig, prog_fig, prog_info

            # Use .input (not .change) so the event only fires on user
            # interaction, NOT when hidden components first render.
            cbm_fault.input(fn=_refresh_all, inputs=_cbm_inputs,
                            outputs=_cbm_outputs)
            cbm_sw.release(fn=_refresh_all, inputs=_cbm_inputs,
                           outputs=_cbm_outputs)
            cbm_sf.release(fn=_refresh_all, inputs=_cbm_inputs,
                           outputs=_cbm_outputs)

            # --- live compute (GPU) -----------------------------------------
            def _on_live_compute(fault, scale, inj_pt, sw, sf):
                if not fault or scale is None or sw is None or sf is None:
                    return _cbm_empty
                cbm_live_compute(fault, scale,
                                 injection_point=int(inj_pt) if inj_pt is not None else None)
                return _refresh_all(fault, sw, sf)

            cbm_live_btn.click(
                fn=_on_live_compute,
                inputs=[cbm_fault, cbm_scale, cbm_inj, cbm_sw, cbm_sf],
                outputs=_cbm_outputs)

        # ============== NAVIGATION ==============
        all_pages = [home_page, realtime_page, charts_page, chats_page, cbm_page]

        def show_page(page_name):
            return (
                gr.update(visible=(page_name == "home")),
                gr.update(visible=(page_name == "realtime")),
                gr.update(visible=(page_name == "charts")),
                gr.update(visible=(page_name == "chats")),
                gr.update(visible=(page_name == "cbm")),
            )

        # Navigation functions for data buttons (now accept time_index from state)
        def goto_chart_bus1(time_index):
            return (*show_page("charts"), "Bus1_Load", create_variable_chart("Bus1_Load", time_index=time_index), create_total_error_chart(time_index=time_index))

        def goto_chart_bus2(time_index):
            return (*show_page("charts"), "Bus2_Load", create_variable_chart("Bus2_Load", time_index=time_index), create_total_error_chart(time_index=time_index))

        def goto_chart_speed(time_index):
            return (*show_page("charts"), "Speed", create_variable_chart("Speed", time_index=time_index), create_total_error_chart(time_index=time_index))

        def goto_chart_position(time_index):
            return (*show_page("charts"), "Latitude", create_variable_chart("Latitude", time_index=time_index), create_total_error_chart(time_index=time_index))

        btn_realtime.click(fn=lambda: show_page("realtime"), outputs=all_pages)
        btn_chats.click(fn=lambda: show_page("chats"), outputs=all_pages)
        btn_cbm.click(fn=lambda: show_page("cbm"), outputs=all_pages).then(
            fn=_refresh_all, inputs=_cbm_inputs, outputs=_cbm_outputs)

        back_btn_rt.click(fn=lambda: show_page("home"), outputs=all_pages)
        back_btn_charts.click(fn=lambda: show_page("realtime"), outputs=all_pages)
        back_btn_chat.click(fn=lambda: show_page("home"), outputs=all_pages)
        back_btn_cbm.click(fn=lambda: show_page("home"), outputs=all_pages)

        def goto_charts_with_state(time_index):
            """Navigate to charts with current time state."""
            return (
                *show_page("charts"),
                create_variable_chart("Bus1_Load", time_index=time_index),
                create_total_error_chart(time_index=time_index)
            )

        view_charts_btn.click(
            fn=goto_charts_with_state,
            inputs=[selected_time_state],
            outputs=[*all_pages, variable_chart, total_error_chart]
        )
        interaction_btn.click(fn=lambda: show_page("realtime"), outputs=all_pages)

        # Data button click handlers - navigate to charts with selected variable and time
        btn_bus1.click(fn=goto_chart_bus1, inputs=[selected_time_state], outputs=[*all_pages, variable_dropdown, variable_chart, total_error_chart])
        btn_bus2.click(fn=goto_chart_bus2, inputs=[selected_time_state], outputs=[*all_pages, variable_dropdown, variable_chart, total_error_chart])
        btn_speed.click(fn=goto_chart_speed, inputs=[selected_time_state], outputs=[*all_pages, variable_dropdown, variable_chart, total_error_chart])
        btn_position.click(fn=goto_chart_position, inputs=[selected_time_state], outputs=[*all_pages, variable_dropdown, variable_chart, total_error_chart])

        # Time slider change handler
        def on_time_slider_change(index):
            """Update ALL displays when slider changes."""
            index = int(index)

            # Update tool executor with selected time (for chat tools)
            if tool_executor:
                tool_executor.set_selected_time(index)

            # Get timestamp for display
            if detector and detector.data_loader._df is not None:
                timestamp = detector.data_loader._df.index[index]
                time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            else:
                time_str = "No data"

            # Update all displays
            html = get_realtime_page_html(time_index=index)
            labels = get_data_button_labels(time_index=index)

            # Return: state, realtime display, time string, buttons, chart
            return (
                index,  # Update state
                html,
                time_str,
                *labels,
            )

        time_slider.change(
            fn=on_time_slider_change,
            inputs=[time_slider],
            outputs=[selected_time_state, realtime_display, time_display, btn_bus1, btn_bus2, btn_speed, btn_position]
        )

        # Refresh button resets to latest data and updates slider
        def refresh_realtime():
            """Refresh to latest data and reset slider."""
            if detector:
                test_info = detector.get_test_data_info()
                latest_index = test_info['end_index']

                # Reset tool executor to latest time
                if tool_executor:
                    tool_executor.set_selected_time(latest_index)

                return (
                    latest_index,  # Update state
                    get_realtime_page_html(time_index=latest_index),
                    latest_index,  # Update slider
                    test_info['end_time'],
                    *get_data_button_labels(time_index=latest_index)
                )
            return initial_index, get_realtime_page_html(), slider_max, "No data", *get_data_button_labels()

        refresh_rt.click(
            fn=refresh_realtime,
            outputs=[selected_time_state, realtime_display, time_slider, time_display, btn_bus1, btn_bus2, btn_speed, btn_position]
        )

    return app


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Vessel Monitoring System")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")

    args = parser.parse_args()

    initialize_system()

    app = create_app()

    # Use Soft theme which supports both light and dark modes
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    )

    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        css=CUSTOM_CSS,
        theme=theme
    )


if __name__ == "__main__":
    main()
