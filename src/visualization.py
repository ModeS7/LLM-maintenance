"""
Visualization utilities for the Gradio UI.
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple, Any
from datetime import datetime, timedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data_loader import MODEL_FEATURES, VARIABLE_GROUPS


# Color scheme
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'warning': '#ffbb33',
    'danger': '#dc3545',
    'healthy': '#28a745',
    'caution': '#ffc107',
    'critical': '#dc3545',
    'reconstruction': '#e377c2',
    'anomaly': '#d62728',
    'grid': '#e5e5e5',
}

# Severity colors
SEVERITY_COLORS = {
    'healthy': COLORS['healthy'],
    'caution': COLORS['caution'],
    'warning': COLORS['warning'],
    'critical': COLORS['critical'],
}


def create_time_series_plot(
    timestamps: List[datetime],
    values: Dict[str, List[float]],
    title: str = "Time Series",
    ylabel: str = "Value",
    anomaly_markers: Optional[List[Tuple[datetime, float]]] = None,
    height: int = 400
) -> go.Figure:
    """
    Create a time series plot with optional anomaly markers.

    Args:
        timestamps: List of timestamps
        values: Dictionary mapping variable names to their values
        title: Plot title
        ylabel: Y-axis label
        anomaly_markers: List of (timestamp, value) tuples for anomaly points
        height: Plot height in pixels

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Add traces for each variable
    for i, (name, vals) in enumerate(values.items()):
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=vals,
            mode='lines',
            name=name,
            line=dict(width=1.5),
        ))

    # Add anomaly markers
    if anomaly_markers:
        anomaly_times = [m[0] for m in anomaly_markers]
        anomaly_values = [m[1] for m in anomaly_markers]

        fig.add_trace(go.Scatter(
            x=anomaly_times,
            y=anomaly_values,
            mode='markers',
            name='Anomaly',
            marker=dict(
                color=COLORS['anomaly'],
                size=10,
                symbol='x',
                line=dict(width=2)
            )
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=ylabel,
        height=height,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=20, t=60, b=40),
    )

    return fig


def create_reconstruction_plot(
    timestamps: List[datetime],
    actual: List[float],
    reconstructed: List[float],
    variable_name: str,
    threshold: Optional[float] = None,
    height: int = 500
) -> go.Figure:
    """
    Create a plot comparing actual vs reconstructed values.

    Args:
        timestamps: List of timestamps
        actual: Actual values
        reconstructed: Reconstructed values
        variable_name: Name of the variable
        threshold: Anomaly threshold for error
        height: Plot height

    Returns:
        Plotly figure with two subplots
    """
    # Compute error
    error = np.abs(np.array(actual) - np.array(reconstructed))

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{variable_name}: Actual vs Reconstructed', 'Reconstruction Error'),
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4]
    )

    # Actual values
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=actual,
        mode='lines',
        name='Actual',
        line=dict(color=COLORS['primary'], width=1.5),
    ), row=1, col=1)

    # Reconstructed values
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=reconstructed,
        mode='lines',
        name='Reconstructed',
        line=dict(color=COLORS['reconstruction'], width=1.5, dash='dash'),
    ), row=1, col=1)

    # Error plot
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=error,
        mode='lines',
        name='Error',
        fill='tozeroy',
        line=dict(color=COLORS['warning'], width=1),
        fillcolor='rgba(255, 187, 51, 0.3)',
    ), row=2, col=1)

    # Threshold line
    if threshold:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=COLORS['danger'],
            annotation_text="Threshold",
            row=2, col=1
        )

    fig.update_layout(
        height=height,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=20, t=80, b=40),
    )

    return fig


def create_anomaly_score_plot(
    timestamps: List[datetime],
    scores: List[float],
    threshold: float,
    severity_levels: Optional[Dict[str, float]] = None,
    height: int = 300
) -> go.Figure:
    """
    Create an anomaly score time series plot.

    Args:
        timestamps: List of timestamps
        scores: Anomaly scores
        threshold: Anomaly threshold
        severity_levels: Dictionary of severity level thresholds
        height: Plot height

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Main score line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=scores,
        mode='lines',
        name='Anomaly Score',
        fill='tozeroy',
        line=dict(color=COLORS['primary'], width=1.5),
        fillcolor='rgba(31, 119, 180, 0.2)',
    ))

    # Threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=COLORS['danger'],
        annotation_text="Anomaly Threshold",
    )

    # Add severity level lines
    if severity_levels:
        for level, value in severity_levels.items():
            if level != 'healthy' and value > 0:
                fig.add_hline(
                    y=value,
                    line_dash="dot",
                    line_color=SEVERITY_COLORS.get(level, COLORS['warning']),
                    annotation_text=level.title(),
                    annotation_position="right",
                )

    fig.update_layout(
        title="Anomaly Score Over Time",
        xaxis_title="Time",
        yaxis_title="Score",
        height=height,
        hovermode='x unified',
        margin=dict(l=60, r=20, t=60, b=40),
    )

    return fig


def create_feature_contribution_plot(
    feature_errors: Dict[str, float],
    top_n: int = 10,
    height: int = 350
) -> go.Figure:
    """
    Create a bar chart of feature contributions to anomaly.

    Args:
        feature_errors: Dictionary mapping feature names to error values
        top_n: Number of top features to show
        height: Plot height

    Returns:
        Plotly figure
    """
    # Sort by error
    sorted_features = sorted(feature_errors.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [f[0] for f in sorted_features]
    errors = [f[1] for f in sorted_features]

    # Color by group
    colors = []
    for name in names:
        color = COLORS['primary']
        for group, vars in VARIABLE_GROUPS.items():
            if name in vars:
                if group == 'electrical':
                    color = '#1f77b4'
                elif group == 'maneuver':
                    color = '#ff7f0e'
                elif group == 'propulsion':
                    color = '#2ca02c'
                elif group == 'ship':
                    color = '#9467bd'
                elif group == 'coordinates':
                    color = '#8c564b'
                break
        colors.append(color)

    fig = go.Figure(go.Bar(
        x=errors,
        y=names,
        orientation='h',
        marker_color=colors,
    ))

    fig.update_layout(
        title="Feature Contribution to Anomaly Score",
        xaxis_title="Reconstruction Error",
        yaxis_title="Variable",
        height=height,
        margin=dict(l=150, r=20, t=60, b=40),
        yaxis=dict(autorange="reversed"),  # Highest at top
    )

    return fig


def create_group_summary_plot(
    group_values: Dict[str, float],
    group_status: Dict[str, str],
    height: int = 250
) -> go.Figure:
    """
    Create a summary bar chart for variable groups.

    Args:
        group_values: Dictionary mapping group names to total values
        group_status: Dictionary mapping group names to health status
        height: Plot height

    Returns:
        Plotly figure
    """
    groups = list(group_values.keys())
    values = list(group_values.values())

    # Color by status
    colors = [SEVERITY_COLORS.get(group_status.get(g, 'healthy'), COLORS['healthy'])
              for g in groups]

    fig = go.Figure(go.Bar(
        x=groups,
        y=values,
        marker_color=colors,
        text=[f"{v:.0f}" for v in values],
        textposition='auto',
    ))

    fig.update_layout(
        title="Power by System",
        xaxis_title="System",
        yaxis_title="Power (kW)",
        height=height,
        margin=dict(l=60, r=20, t=60, b=40),
    )

    return fig


def create_status_indicator(
    status: str,
    score: float,
    size: str = "large"
) -> str:
    """
    Create an HTML status indicator.

    Args:
        status: Status level ('healthy', 'caution', 'warning', 'critical')
        score: Anomaly score
        size: 'large' or 'small'

    Returns:
        HTML string
    """
    color = SEVERITY_COLORS.get(status, COLORS['healthy'])

    if size == "large":
        indicator_size = "20px"
        font_size = "24px"
    else:
        indicator_size = "12px"
        font_size = "14px"

    html = f"""
    <div style="display: flex; align-items: center; gap: 10px;">
        <span style="
            display: inline-block;
            width: {indicator_size};
            height: {indicator_size};
            border-radius: 50%;
            background-color: {color};
            box-shadow: 0 0 8px {color};
        "></span>
        <span style="font-size: {font_size}; font-weight: bold; color: {color};">
            {status.upper()}
        </span>
        <span style="font-size: {font_size}; color: #666;">
            (Score: {score:.4f})
        </span>
    </div>
    """

    return html


def create_variable_card(
    group: str,
    value: float,
    unit: str,
    status: str = "healthy"
) -> str:
    """
    Create an HTML card for a variable group.

    Args:
        group: Group name
        value: Total value
        unit: Unit string
        status: Health status

    Returns:
        HTML string
    """
    color = SEVERITY_COLORS.get(status, COLORS['healthy'])

    html = f"""
    <div style="
        padding: 15px;
        border-radius: 8px;
        background: linear-gradient(135deg, #f5f5f5, #e5e5e5);
        border-left: 4px solid {color};
        text-align: center;
    ">
        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">
            {group.title()}
        </div>
        <div style="font-size: 24px; font-weight: bold; color: #333;">
            {value:.1f}
        </div>
        <div style="font-size: 12px; color: #888;">
            {unit}
        </div>
        <div style="margin-top: 8px;">
            <span style="
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background-color: {color};
            "></span>
        </div>
    </div>
    """

    return html


def format_timestamp(ts: datetime) -> str:
    """Format timestamp for display."""
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def format_value(value: float, precision: int = 2) -> str:
    """Format numeric value for display."""
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    elif abs(value) >= 100:
        return f"{value:.1f}"
    else:
        return f"{value:.{precision}f}"
