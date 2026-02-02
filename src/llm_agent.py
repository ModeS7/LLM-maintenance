"""
LLM Agent with tool calling for vessel monitoring.
"""
import json
from typing import Generator, List, Dict, Optional, Any
from datetime import datetime

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from .tools import TOOLS, ToolExecutor, format_tool_result


SYSTEM_PROMPT = """/no_think
You are an AI assistant for offshore vessel monitoring. You help operators understand vessel power systems, detect anomalies, and provide maintenance insights.

You have access to tools to query real-time vessel data. ALWAYS use tools to get actual data - never make up numbers. When reporting anomalies, explain which variables are contributing and suggest possible causes.

Variable Groups:
- Electrical Load: Bus1/Bus2 Load and Available Load (main power distribution)
- Maneuver: Bow thrusters (1-3), Stern thrusters (1-2) - used for positioning
- Propulsion: Main propulsion drives and engines (PS=Port Side, SB=Starboard)
- Ship Variables: Draft (fore/aft), Speed
- Coordinates: Latitude, Longitude

When analyzing anomalies:
1. First get the current vessel status
2. Check which variables are contributing most to the anomaly
3. Explain what the deviation means operationally
4. Suggest potential causes (equipment issues, operational changes, environmental factors)

Be concise but informative. Use maritime terminology appropriately."""


class VesselMaintenanceAgent:
    """LLM agent for vessel maintenance assistance."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        tool_executor: Optional[ToolExecutor] = None
    ):
        """
        Initialize the agent.

        Args:
            model: Ollama model name to use
            tool_executor: ToolExecutor instance for executing tools
        """
        self.model = model
        self.tool_executor = tool_executor
        self.conversation_history: List[Dict] = []
        self.current_context: Dict = {}

        if not OLLAMA_AVAILABLE:
            raise ImportError("ollama package not installed. Run: pip install ollama")

    def set_tool_executor(self, tool_executor: ToolExecutor):
        """Set the tool executor."""
        self.tool_executor = tool_executor

    def set_context(
        self,
        current_time: Optional[datetime] = None,
        selected_variable: Optional[str] = None
    ):
        """Set context for the conversation."""
        self.current_context = {
            'current_time': str(current_time) if current_time else str(datetime.now()),
            'selected_variable': selected_variable
        }

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute tool calls and return results."""
        results = []

        for tool_call in tool_calls:
            function = tool_call.get('function', {})
            name = function.get('name', '')
            arguments = function.get('arguments', {})

            # Parse arguments if string
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            # Execute tool
            if self.tool_executor:
                result = self.tool_executor.execute(name, arguments)
            else:
                result = {"error": "Tool executor not configured"}

            results.append({
                'tool_name': name,
                'arguments': arguments,
                'result': result
            })

        return results

    def chat(self, message: str) -> str:
        """
        Send a message and get a response.

        Args:
            message: User message

        Returns:
            Assistant response
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })

        # Build messages with system prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + self.conversation_history

        # Initial request with tools
        response = ollama.chat(
            model=self.model,
            messages=messages,
            tools=TOOLS
        )

        assistant_message = response.get('message', {})
        tool_calls = assistant_message.get('tool_calls', [])

        # Handle tool calls iteratively
        max_iterations = 5
        iteration = 0

        while tool_calls and iteration < max_iterations:
            iteration += 1

            # Execute tools
            tool_results = self._execute_tool_calls(tool_calls)

            # Add assistant message with tool calls
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message.get('content', ''),
                "tool_calls": tool_calls
            })

            # Add tool results
            for tr in tool_results:
                self.conversation_history.append({
                    "role": "tool",
                    "content": json.dumps(tr['result'])
                })

            # Get next response
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ] + self.conversation_history

            response = ollama.chat(
                model=self.model,
                messages=messages,
                tools=TOOLS
            )

            assistant_message = response.get('message', {})
            tool_calls = assistant_message.get('tool_calls', [])

        # Final response
        final_content = assistant_message.get('content', '')

        # Clean up any thinking tags that might leak through (Qwen3 issue)
        import re
        final_content = re.sub(r'<think>.*?</think>', '', final_content, flags=re.DOTALL)
        final_content = final_content.strip()

        # Add final assistant message to history
        self.conversation_history.append({
            "role": "assistant",
            "content": final_content
        })

        return final_content

    def chat_stream(self, message: str) -> Generator[str, None, None]:
        """
        Send a message and stream the response.

        Args:
            message: User message

        Yields:
            Response chunks
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })

        # Build messages with system prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + self.conversation_history

        # First, do a non-streaming call to handle tool calls
        response = ollama.chat(
            model=self.model,
            messages=messages,
            tools=TOOLS
        )

        assistant_message = response.get('message', {})
        tool_calls = assistant_message.get('tool_calls', [])

        # Handle tool calls
        max_iterations = 5
        iteration = 0

        while tool_calls and iteration < max_iterations:
            iteration += 1

            # Yield tool execution status
            yield f"\n[Executing tools...]\n"

            # Execute tools
            tool_results = self._execute_tool_calls(tool_calls)

            # Yield tool results summary
            for tr in tool_results:
                yield f"  - {tr['tool_name']}: "
                if 'error' in tr['result']:
                    yield f"Error - {tr['result']['error']}\n"
                else:
                    yield "OK\n"

            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message.get('content', ''),
                "tool_calls": tool_calls
            })

            for tr in tool_results:
                self.conversation_history.append({
                    "role": "tool",
                    "content": json.dumps(tr['result'])
                })

            # Get next response
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ] + self.conversation_history

            response = ollama.chat(
                model=self.model,
                messages=messages,
                tools=TOOLS
            )

            assistant_message = response.get('message', {})
            tool_calls = assistant_message.get('tool_calls', [])

        # Now stream the final response
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + self.conversation_history

        # Stream final response
        full_response = ""
        for chunk in ollama.chat(
            model=self.model,
            messages=messages,
            stream=True
        ):
            content = chunk.get('message', {}).get('content', '')
            if content:
                full_response += content
                yield content

        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": full_response
        })

    def get_quick_response(self, query_type: str) -> str:
        """
        Get a quick response for common queries.

        Args:
            query_type: One of 'status', 'anomalies', 'power'

        Returns:
            Response string
        """
        queries = {
            'status': "What is the current vessel status? Give me a brief overview.",
            'anomalies': "Have there been any anomalies in the last 24 hours?",
            'power': "What is the current power distribution across the vessel?"
        }

        query = queries.get(query_type, queries['status'])
        return self.chat(query)


class MockVesselMaintenanceAgent:
    """Mock agent for when Ollama is not available."""

    def __init__(self, tool_executor: Optional[ToolExecutor] = None):
        self.tool_executor = tool_executor
        self.conversation_history: List[Dict] = []

    def set_tool_executor(self, tool_executor: ToolExecutor):
        self.tool_executor = tool_executor

    def set_context(self, **kwargs):
        pass

    def clear_history(self):
        self.conversation_history = []

    def chat(self, message: str) -> str:
        """Provide responses using tools directly."""
        message_lower = message.lower()

        # Add to history
        self.conversation_history.append({"role": "user", "content": message})

        response = ""

        if self.tool_executor:
            if any(word in message_lower for word in ['status', 'how is', 'overview']):
                result = self.tool_executor.execute('get_vessel_status', {})
                response = f"**Current Vessel Status**\n\n{format_tool_result(result)}"

            elif any(word in message_lower for word in ['anomal', 'issue', 'problem']):
                result = self.tool_executor.execute('get_anomaly_history', {'hours': 24})
                response = f"**Anomaly Report**\n\n{format_tool_result(result)}"

            elif 'electrical' in message_lower:
                result = self.tool_executor.execute('get_variable_readings', {'group': 'electrical'})
                response = f"**Electrical System**\n\n{format_tool_result(result)}"

            elif any(word in message_lower for word in ['maneuver', 'thruster']):
                result = self.tool_executor.execute('get_variable_readings', {'group': 'maneuver'})
                response = f"**Maneuvering Systems**\n\n{format_tool_result(result)}"

            elif 'propulsion' in message_lower or 'engine' in message_lower:
                result = self.tool_executor.execute('get_variable_readings', {'group': 'propulsion'})
                response = f"**Propulsion System**\n\n{format_tool_result(result)}"

            elif 'power' in message_lower:
                status = self.tool_executor.execute('get_vessel_status', {})
                response = (f"**Power Overview**\n\n"
                           f"Total Power: {status.get('total_power', 0):.0f} kW\n"
                           f"Bus 1 Load: {status.get('bus1_load', 0):.0f} kW\n"
                           f"Bus 2 Load: {status.get('bus2_load', 0):.0f} kW\n"
                           f"Maneuver Power: {status.get('maneuver_power', 0):.0f} kW\n"
                           f"Propulsion Power: {status.get('propulsion_power', 0):.0f} kW")

            else:
                result = self.tool_executor.execute('get_vessel_status', {})
                response = f"Here's the current vessel status:\n\n{format_tool_result(result)}"
        else:
            response = "Tool executor not configured. Please check the system setup."

        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def chat_stream(self, message: str) -> Generator[str, None, None]:
        """Stream response word by word."""
        response = self.chat(message)
        # Remove the last history entry since chat added it
        self.conversation_history.pop()

        words = response.split(' ')
        full_response = ""
        for word in words:
            chunk = word + ' '
            full_response += chunk
            yield chunk

        self.conversation_history.append({"role": "assistant", "content": full_response})

    def get_quick_response(self, query_type: str) -> str:
        queries = {
            'status': "What is the current vessel status?",
            'anomalies': "Show me recent anomalies",
            'power': "What is the power distribution?"
        }
        return self.chat(queries.get(query_type, queries['status']))


def create_agent(
    model: str = "qwen3:8b",
    tool_executor: Optional[ToolExecutor] = None
) -> Any:
    """
    Create an appropriate agent based on availability.

    Args:
        model: Ollama model name
        tool_executor: ToolExecutor instance

    Returns:
        Agent instance (real or mock)
    """
    if OLLAMA_AVAILABLE:
        try:
            # Test if ollama is running
            ollama.list()
            return VesselMaintenanceAgent(model=model, tool_executor=tool_executor)
        except Exception:
            pass

    return MockVesselMaintenanceAgent(tool_executor=tool_executor)
