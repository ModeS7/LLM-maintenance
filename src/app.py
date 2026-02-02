"""
Gradio UI for Vessel Monitoring System - Matching PowerPoint Design.
"""
import gradio as gr
import numpy as np
from pathlib import Path
from typing import Optional, Any
import plotly.graph_objects as go

from .data_loader import VesselDataLoader, MODEL_FEATURES
from .inference import AnomalyDetector
from .tools import ToolExecutor
from .llm_agent import create_agent


# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "Data_Pwr_All_S5.txt"
MODEL_PATH = BASE_DIR / "models" / "autoencoder.pt"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

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


def get_engine_html():
    """Generate HTML for engine display with gauges."""
    global ENGINE_IMAGE_DATA

    # Load image on first use
    if ENGINE_IMAGE_DATA is None:
        ENGINE_IMAGE_DATA = get_engine_image_base64()

    if detector:
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
        alerts_html += f'<div style="background:#5c2626; border-left:4px solid #f87171; padding:8px 15px; border-radius:5px; font-size:13px; color:#fca5a5; white-space: nowrap;">{alert}</div>'

    if not alerts_html:
        alerts_html = '<div style="background:#1e4035; border-left:4px solid #4ade80; padding:8px 15px; border-radius:5px; color:#86efac; font-size:14px; font-weight: 500;">All systems normal</div>'

    # Use base64 image or fallback
    img_data, img_mime = ENGINE_IMAGE_DATA if ENGINE_IMAGE_DATA else (None, None)
    if img_data:
        img_src = f"data:{img_mime};base64,{img_data}"
    else:
        img_src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='380' height='220'%3E%3Crect fill='%23243447' width='380' height='220' rx='10'/%3E%3Ctext x='190' y='110' fill='%23a8b8c8' text-anchor='middle' font-size='20'%3EMarine Diesel Engine%3C/text%3E%3C/svg%3E"

    return f'''
    <div style="background: #243447; padding: 25px 30px; border-radius: 12px; border: 1px solid #3d5a73;">
        <div style="display: flex; justify-content: center; align-items: center; gap: 30px;">

            <!-- Left Gauge - RPM -->
            <div style="display: flex; flex-direction: column; align-items: center; flex-shrink: 0;">
                <div style="background: linear-gradient(135deg, #1e4d5c, #2d6b7a); border-radius: 50%; width: 130px; height: 130px; display: flex; flex-direction: column; align-items: center; justify-content: center; font-weight: bold; box-shadow: 0 6px 25px rgba(0,0,0,0.4); border: 4px solid #4a9eff;">
                    <span style="font-size: 32px; color: #4ade80;">{rpm}</span>
                    <span style="font-size: 14px; color: #a8b8c8;">RPM</span>
                </div>
            </div>

            <!-- Center - Engine Image -->
            <div style="flex: 1; display: flex; flex-direction: column; align-items: center; max-width: 700px;">
                <img src="{img_src}"
                     style="width: 100%; height: auto; border-radius: 12px; box-shadow: 0 8px 30px rgba(0,0,0,0.5); border: 2px solid #3d5a73;"
                     alt="Marine Diesel Engine">
                <!-- Alerts below image -->
                <div style="margin-top: 15px; display: flex; gap: 10px; flex-wrap: wrap; justify-content: center;">
                    {alerts_html}
                </div>
            </div>

            <!-- Right Gauge - Power -->
            <div style="display: flex; flex-direction: column; align-items: center; flex-shrink: 0;">
                <div style="background: linear-gradient(135deg, #1e4d5c, #2d6b7a); border-radius: 50%; width: 130px; height: 130px; display: flex; flex-direction: column; align-items: center; justify-content: center; font-weight: bold; box-shadow: 0 6px 25px rgba(0,0,0,0.4); border: 4px solid #4a9eff;">
                    <span style="font-size: 32px; color: #4ade80;">{power_mw:.1f}</span>
                    <span style="font-size: 14px; color: #a8b8c8;">MW</span>
                </div>
            </div>
        </div>
    </div>
    '''


def get_status_buttons_html():
    """Generate status buttons for OIL, WATER, AIR, FUEL."""
    if detector:
        health = detector.get_feature_health()
        oil_ok = health.get('Main_Prop_PS_ME1_Power', 'healthy') == 'healthy'
        water_ok = health.get('Bus1_Load', 'healthy') == 'healthy'
        air_ok = health.get('Speed', 'healthy') == 'healthy'
        fuel_ok = health.get('Main_Prop_PS_Drive_Power', 'healthy') == 'healthy'
    else:
        oil_ok = water_ok = air_ok = fuel_ok = True

    def btn_style(ok):
        if ok:
            bg = "linear-gradient(135deg, #059669, #10b981)"
            border = "#34d399"
        else:
            bg = "linear-gradient(135deg, #dc2626, #ef4444)"
            border = "#f87171"
        return f"background: {bg}; color: white; font-weight: 600; padding: 14px 35px; border: 2px solid {border}; border-radius: 10px; font-size: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); cursor: pointer; transition: transform 0.2s;"

    return f'''
    <div style="display: flex; justify-content: center; gap: 15px; margin: 20px 0;">
        <button style="{btn_style(oil_ok)}">🛢️ OIL</button>
        <button style="{btn_style(water_ok)}">💧 WATER</button>
        <button style="{btn_style(air_ok)}">💨 AIR</button>
        <button style="{btn_style(fuel_ok)}">⛽ FUEL</button>
    </div>
    '''


def get_variables_html():
    """Generate variable display boxes."""
    if detector:
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
            <div style="background: #1a2332; border: 1px solid #3d5a73; border-radius: 12px; padding: 15px 20px; margin-bottom: 8px;">
                <div style="font-weight: 600; font-size: 18px; color: #4ade80;">{value}</div>
            </div>
            <div style="font-size: 13px; color: #a8b8c8;">{name}</div>
        </div>
        '''

    return f'<div style="display: flex; justify-content: center; margin-top: 20px; flex-wrap: wrap; gap: 10px;">{boxes_html}</div>'


def get_realtime_page_html():
    """Generate complete real-time page HTML."""
    return f'''
    <div style="background: #243447; padding: 25px; border-radius: 12px; border: 1px solid #3d5a73;">
        {get_engine_html()}
        {get_status_buttons_html()}
        {get_variables_html()}
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
/* ========== BASE STYLES ========== */
.gradio-container {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d5a7b 50%, #1e3a5f 100%) !important;
    min-height: 100vh;
}

/* ========== DARK MODE VARIABLES ========== */
:root {
    --bg-primary: #1a2332;
    --bg-secondary: #243447;
    --bg-card: #2d4156;
    --text-primary: #e8eef4;
    --text-secondary: #a8b8c8;
    --accent-blue: #4a9eff;
    --accent-green: #4ade80;
    --accent-orange: #fb923c;
    --accent-red: #f87171;
    --accent-purple: #a78bfa;
    --border-color: #3d5a73;
}

/* ========== HOME PAGE ========== */
.home-btn {
    background: linear-gradient(135deg, var(--accent-purple) 0%, #8b5cf6 100%) !important;
    color: white !important;
    font-size: 20px !important;
    font-weight: 600 !important;
    padding: 25px 50px !important;
    border-radius: 12px !important;
    border: none !important;
    min-width: 400px !important;
    box-shadow: 0 4px 20px rgba(167, 139, 250, 0.3) !important;
    transition: all 0.3s ease !important;
}
.home-btn:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 6px 25px rgba(167, 139, 250, 0.4) !important;
}

/* ========== NAVIGATION ========== */
.back-btn {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    font-size: 24px !important;
    padding: 8px 16px !important;
    border-radius: 8px !important;
    border: 1px solid var(--border-color) !important;
}
.back-btn:hover {
    background: var(--bg-secondary) !important;
}

.interaction-btn {
    background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%) !important;
    color: #1a1a1a !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    border-radius: 8px !important;
}

/* ========== CARDS & PANELS ========== */
.card-panel {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    border: 1px solid var(--border-color) !important;
}

/* ========== CHAT STYLES ========== */
.chat-sidebar {
    background: var(--bg-secondary) !important;
    border-radius: 12px !important;
    padding: 15px !important;
    border: 1px solid var(--border-color) !important;
}

/* Chatbot container */
.chatbot {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border-color) !important;
}

/* Chat messages */
.message {
    border-radius: 12px !important;
    padding: 12px 16px !important;
}

/* User message */
.user-message, .message.user {
    background: var(--accent-blue) !important;
    color: white !important;
}

/* Bot message */
.bot-message, .message.bot {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
}

/* ========== BUTTONS ========== */
.primary-btn, button.primary {
    background: linear-gradient(135deg, var(--accent-blue) 0%, #3b82f6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.primary-btn:hover, button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(74, 158, 255, 0.3) !important;
}

.secondary-btn, button.secondary {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
}
.secondary-btn:hover, button.secondary:hover {
    background: var(--bg-secondary) !important;
}

/* Quick question buttons */
.quick-btn {
    background: var(--bg-secondary) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 20px !important;
    font-size: 13px !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease !important;
}
.quick-btn:hover {
    background: var(--accent-blue) !important;
    color: white !important;
    border-color: var(--accent-blue) !important;
}

/* ========== INPUTS ========== */
input, textarea, .input-text {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
}
input:focus, textarea:focus {
    border-color: var(--accent-blue) !important;
    outline: none !important;
}

/* ========== RADIO BUTTONS (Chat list) ========== */
.radio-group label {
    background: transparent !important;
    color: var(--text-secondary) !important;
    padding: 10px 12px !important;
    border-radius: 8px !important;
    margin: 4px 0 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
.radio-group label:hover {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
}
.radio-group input:checked + label {
    background: var(--accent-blue) !important;
    color: white !important;
}

/* ========== CHARTS ========== */
.plot-container {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    padding: 15px !important;
    border: 1px solid var(--border-color) !important;
}

/* ========== STATUS INDICATORS ========== */
.status-healthy { color: var(--accent-green) !important; }
.status-warning { color: var(--accent-orange) !important; }
.status-critical { color: var(--accent-red) !important; }

/* ========== HEADINGS ========== */
h1, h2, h3, h4 {
    color: var(--text-primary) !important;
}

/* ========== SCROLLBAR ========== */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: var(--bg-primary);
}
::-webkit-scrollbar-thumb {
    background: var(--border-color);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--accent-blue);
}

/* ========== DYNAMIC HEIGHT CHAT ========== */
.chat-fullpage {
    height: calc(100vh - 80px) !important;
    display: flex !important;
    flex-direction: column !important;
}

/* Target the chatbot inside ChatInterface */
.chat-fullpage .chatbot {
    height: calc(100vh - 350px) !important;
    min-height: 300px !important;
}

/* ChatInterface with save_history sidebar */
.chat-fullpage .chat-interface-row {
    flex: 1 !important;
    min-height: 0 !important;
}

/* Chat history sidebar styling */
.chat-fullpage .sidebar,
.chat-fullpage [class*="sidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border-color) !important;
}

.chat-fullpage .sidebar button,
.chat-fullpage [class*="sidebar"] button {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border: none !important;
    text-align: left !important;
    padding: 12px !important;
    border-radius: 8px !important;
    margin: 4px 8px !important;
}

.chat-fullpage .sidebar button:hover,
.chat-fullpage [class*="sidebar"] button:hover {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
}

.chat-fullpage .sidebar button.selected,
.chat-fullpage [class*="sidebar"] button[aria-selected="true"] {
    background: var(--accent-blue) !important;
    color: white !important;
}

/* New chat button */
.chat-fullpage button[class*="new-chat"],
.chat-fullpage [class*="new"] {
    background: var(--accent-blue) !important;
    color: white !important;
    border-radius: 8px !important;
    margin: 8px !important;
}

/* ========== RESPONSIVE ========== */
@media (max-width: 768px) {
    .home-btn {
        min-width: 280px !important;
        font-size: 18px !important;
        padding: 20px 30px !important;
    }
    .chat-fullpage .chatbot {
        height: calc(100vh - 300px) !important;
    }
}
"""


def create_app() -> gr.Blocks:
    """Create the Gradio application."""

    with gr.Blocks(title="Vessel Monitoring System") as app:

        # ============== HOME PAGE ==============
        with gr.Column(visible=True) as home_page:
            gr.HTML('''
                <div style="background: linear-gradient(135deg, #243447 0%, #2d4156 100%);
                            padding: 60px 40px; border-radius: 16px; min-height: 480px;
                            border: 1px solid #3d5a73;">
                    <h1 style="text-align: center; color: #e8eef4; margin-bottom: 15px; font-size: 36px; font-weight: 600;">
                        Vessel Monitoring System
                    </h1>
                    <p style="text-align: center; color: #a8b8c8; margin-bottom: 50px; font-size: 16px;">
                        AI-powered monitoring for offshore vessel operations
                    </p>
                    <div style="display: flex; flex-direction: column; align-items: center; gap: 20px;">
            ''')
            btn_realtime = gr.Button("REAL TIME MONITORING", elem_classes=["home-btn"])
            btn_chats = gr.Button("AI CHAT ASSISTANT", elem_classes=["home-btn"])
            gr.HTML('''
                    </div>
                    <p style="text-align: center; color: #6b7c8c; margin-top: 40px; font-size: 13px;">
                        M/S Olympic Hera • Offshore Construction Vessel
                    </p>
                </div>
            ''')

        # ============== REAL TIME PAGE ==============
        with gr.Column(visible=False) as realtime_page:
            with gr.Row():
                back_btn_rt = gr.Button("Back", elem_classes=["back-btn"], scale=0, min_width=100)
                gr.HTML('<div style="flex:1;"><h2 style="text-align:center; color:#e8eef4; margin:0;">Real Time Monitoring</h2></div>')
                view_charts_btn = gr.Button("Charts", variant="secondary")

            realtime_display = gr.HTML(value=get_realtime_page_html)
            refresh_rt = gr.Button("Refresh Data", variant="primary")

        # ============== CHARTS PAGE ==============
        with gr.Column(visible=False) as charts_page:
            with gr.Row():
                back_btn_charts = gr.Button("Back", elem_classes=["back-btn"], scale=0, min_width=100)
                gr.HTML('<div style="flex:1;"><h2 style="text-align:center; color:#e8eef4; margin:0;">Analytics & Charts</h2></div>')
                interaction_btn = gr.Button("Live View", elem_classes=["interaction-btn"])

            gr.HTML('<div style="background: #243447; padding: 20px; border-radius: 12px; border: 1px solid #3d5a73;">')
            anomaly_chart = gr.Plot(value=create_anomaly_chart, label="Time Series with Anomaly Detection")
            comparison_chart = gr.Plot(value=create_comparison_chart, label="Actual vs Reconstructed (AI Prediction)")
            gr.HTML('</div>')

        # ============== CHATS PAGE ==============
        with gr.Column(visible=False, elem_classes=["chat-fullpage"]) as chats_page:
            with gr.Row():
                back_btn_chat = gr.Button("Back", elem_classes=["back-btn"], scale=0, min_width=100)
                gr.HTML('<div style="flex:1;"><h2 style="text-align:center; color:#e8eef4; margin:0;">AI Assistant</h2></div>')
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

            # Use built-in ChatInterface (no custom chatbot to avoid save_history bug)
            gr.ChatInterface(
                fn=chat_fn,
                title=None,
                description="Ask me about vessel status, power systems, anomalies, and more.",
                examples=[
                    "What is the current vessel status?",
                    "Show me the electrical system readings",
                    "Are there any anomalies detected?",
                    "What is the propulsion power output?",
                    "Show the current speed and position",
                ],
                save_history=True,
                fill_height=True,
            )

        # ============== NAVIGATION ==============
        all_pages = [home_page, realtime_page, charts_page, chats_page]

        def show_page(page_name):
            return (
                gr.update(visible=(page_name == "home")),
                gr.update(visible=(page_name == "realtime")),
                gr.update(visible=(page_name == "charts")),
                gr.update(visible=(page_name == "chats")),
            )

        btn_realtime.click(fn=lambda: show_page("realtime"), outputs=all_pages)
        btn_chats.click(fn=lambda: show_page("chats"), outputs=all_pages)

        back_btn_rt.click(fn=lambda: show_page("home"), outputs=all_pages)
        back_btn_charts.click(fn=lambda: show_page("realtime"), outputs=all_pages)
        back_btn_chat.click(fn=lambda: show_page("home"), outputs=all_pages)

        view_charts_btn.click(fn=lambda: show_page("charts"), outputs=all_pages)
        interaction_btn.click(fn=lambda: show_page("realtime"), outputs=all_pages)

        refresh_rt.click(fn=get_realtime_page_html, outputs=realtime_display)

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
    # Create dark theme
    dark_theme = gr.themes.Base(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    ).set(
        body_background_fill="#1a2332",
        body_background_fill_dark="#1a2332",
        block_background_fill="#243447",
        block_background_fill_dark="#243447",
        block_border_color="#3d5a73",
        block_border_color_dark="#3d5a73",
        input_background_fill="#1a2332",
        input_background_fill_dark="#1a2332",
        button_primary_background_fill="#4a9eff",
        button_primary_background_fill_dark="#4a9eff",
        button_secondary_background_fill="#2d4156",
        button_secondary_background_fill_dark="#2d4156",
    )

    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        css=CUSTOM_CSS,
        theme=dark_theme
    )


if __name__ == "__main__":
    main()
