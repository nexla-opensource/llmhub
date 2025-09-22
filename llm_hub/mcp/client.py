from __future__ import annotations
from typing import Dict, Any, List, Optional
from mcp import ClientSession, StdioServerParameters, connect_stdio

class MCPClientManager:
    """
    Minimal MCP client wrapper. You can connect to MCP servers and list/call tools,
    then surface those tool signatures as ToolSpecs in LLM Hub.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, ClientSession] = {}

    async def connect_stdio(self, name: str, command: str, args: Optional[List[str]] = None) -> None:
        conn = await connect_stdio(StdioServerParameters(command=command, args=args or []))
        self._sessions[name] = conn.session

    async def list_tools(self, name: str) -> List[Dict[str, Any]]:
        sess = self._sessions[name]
        tools = await sess.list_tools()
        return tools

    async def call_tool(self, name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        sess = self._sessions[name]
        result = await sess.call_tool(tool_name, arguments=arguments)
        return result
