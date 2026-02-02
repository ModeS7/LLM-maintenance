"""
Tool definitions for LLM agent.
"""
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime


# Tool definitions in Ollama format
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_vessel_status",
            "description": "Get current vessel operational status including power, speed, position, and anomaly score. Use this to check overall vessel health.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_variable_readings",
            "description": "Get current readings for a specific variable group. Groups: 'electrical' (bus loads), 'maneuver' (bow/stern thrusters), 'propulsion' (main engines), 'ship' (draft, speed), 'coordinates' (position)",
            "parameters": {
                "type": "object",
                "properties": {
                    "group": {
                        "type": "string",
                        "description": "Variable group to query",
                        "enum": ["electrical", "maneuver", "propulsion", "ship", "coordinates"]
                    }
                },
                "required": ["group"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_anomaly_history",
            "description": "Get recent anomaly detections. Returns list of detected anomalies with timestamps and severity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hours": {
                        "type": "integer",
                        "description": "Number of hours to look back (default: 24)",
                        "default": 24
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_variable_chart_data",
            "description": "Get time series data for charting a specific variable. Returns timestamps, actual values, and reconstructed values from the model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "variable": {
                        "type": "string",
                        "description": "Variable name to chart (e.g., 'Bus1_Load', 'Speed', 'BowThr1_Power')"
                    },
                    "hours": {
                        "type": "number",
                        "description": "Number of hours of data to retrieve (default: 1)",
                        "default": 1
                    }
                },
                "required": ["variable"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_anomaly",
            "description": "Analyze a specific anomaly event in detail. Provides information about contributing factors and affected systems.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp": {
                        "type": "string",
                        "description": "ISO format timestamp of the anomaly to analyze (e.g., '2011-12-15T10:30:00')"
                    }
                },
                "required": ["timestamp"]
            }
        }
    }
]


class ToolExecutor:
    """Executes tools using the anomaly detector."""

    def __init__(self, detector):
        """
        Initialize with an AnomalyDetector instance.

        Args:
            detector: AnomalyDetector instance
        """
        self.detector = detector

        # Map tool names to methods
        self._tools: Dict[str, Callable] = {
            "get_vessel_status": self._get_vessel_status,
            "get_variable_readings": self._get_variable_readings,
            "get_anomaly_history": self._get_anomaly_history,
            "get_variable_chart_data": self._get_variable_chart_data,
            "analyze_anomaly": self._analyze_anomaly,
        }

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict:
        """
        Execute a tool with given arguments.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if tool_name not in self._tools:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return self._tools[tool_name](**arguments)
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}

    def _get_vessel_status(self) -> Dict:
        """Get current vessel status."""
        return self.detector.get_current_status()

    def _get_variable_readings(self, group: str) -> Dict:
        """Get readings for a variable group."""
        return self.detector.get_variable_readings(group)

    def _get_anomaly_history(self, hours: int = 24) -> Dict:
        """Get anomaly history."""
        anomalies = self.detector.get_anomaly_history(hours)
        return {
            "hours": hours,
            "count": len(anomalies),
            "anomalies": anomalies
        }

    def _get_variable_chart_data(self, variable: str, hours: float = 1.0) -> Dict:
        """Get chart data for a variable."""
        return self.detector.get_reconstruction_comparison(variable, hours)

    def _analyze_anomaly(self, timestamp: str) -> Dict:
        """Analyze a specific anomaly."""
        return self.detector.analyze_anomaly(timestamp)


def format_tool_result(result: Dict) -> str:
    """
    Format tool result as a human-readable string.

    Args:
        result: Tool execution result

    Returns:
        Formatted string
    """
    if "error" in result:
        return f"Error: {result['error']}"

    lines = []

    # Handle different result types
    if "anomaly_score" in result:
        # Vessel status
        lines.append(f"Timestamp: {result.get('timestamp', 'N/A')}")
        lines.append(f"Status: {result.get('severity', 'unknown').upper()}")
        lines.append(f"Anomaly Score: {result.get('anomaly_score', 0):.4f}")
        lines.append(f"Speed: {result.get('speed', 0):.1f} knots")
        lines.append(f"Total Power: {result.get('total_power', 0):.0f} kW")
        lines.append(f"Position: {result.get('latitude', 0):.4f}°N, {result.get('longitude', 0):.4f}°W")

        if result.get('top_contributors'):
            lines.append("\nTop Contributing Variables:")
            for var, error in result['top_contributors'][:3]:
                lines.append(f"  - {var}: {error:.4f}")

    elif "readings" in result:
        # Variable readings
        lines.append(f"Group: {result.get('group', 'unknown').title()}")
        lines.append(f"Timestamp: {result.get('timestamp', 'N/A')}")
        lines.append("\nReadings:")
        for var, value in result['readings'].items():
            if var != 'total':
                unit = 'kW' if 'Power' in var or 'Load' in var else ''
                lines.append(f"  - {var}: {value:.2f} {unit}")
        if 'total' in result['readings']:
            lines.append(f"\nTotal: {result['readings']['total']:.2f} kW")

    elif "anomalies" in result:
        # Anomaly history
        lines.append(f"Anomaly History (last {result.get('hours', 24)} hours)")
        lines.append(f"Total anomalies: {result.get('count', 0)}")

        if result['anomalies']:
            lines.append("\nRecent anomalies:")
            for anomaly in result['anomalies'][-5:]:  # Show last 5
                lines.append(f"  - {anomaly['timestamp']}: "
                           f"Score={anomaly['anomaly_score']:.4f}, "
                           f"Severity={anomaly['severity']}")

    elif "analysis" in result:
        # Anomaly analysis
        lines.append(f"Anomaly Analysis")
        lines.append(f"Requested Time: {result.get('requested_timestamp', 'N/A')}")
        lines.append(f"Peak Time: {result.get('peak_anomaly_time', 'N/A')}")
        lines.append(f"Peak Score: {result.get('peak_anomaly_score', 0):.4f}")
        lines.append(f"Severity: {result.get('severity', 'unknown')}")
        lines.append(f"\nAnalysis: {result.get('analysis', 'N/A')}")

    elif "actual" in result:
        # Chart data
        lines.append(f"Variable: {result.get('variable', 'N/A')}")
        lines.append(f"Data points: {len(result.get('actual', []))}")

        if result.get('actual'):
            actual = result['actual']
            lines.append(f"Latest value: {actual[-1]:.2f}")
            lines.append(f"Min: {min(actual):.2f}")
            lines.append(f"Max: {max(actual):.2f}")

    else:
        # Generic formatting
        for key, value in result.items():
            lines.append(f"{key}: {value}")

    return "\n".join(lines)
