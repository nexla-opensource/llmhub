import asyncio
from llm_hub import LLMHub, HubConfig, OpenAIConfig, ToolSpec
from llm_hub.mcp.client import MCPClientManager

async def main():
    # Setup hub
    hub = LLMHub(HubConfig(
        openai=OpenAIConfig(api_key="OPENAI_API_KEY"),
    ))

    # Setup MCP (optional)
    mcp = MCPClientManager()
    await mcp.connect_stdio("files", command="my-mcp-server-binary")
    mcp_tools = await mcp.list_tools("files")
    print(f"MCP tools: {mcp_tools}")

    # Define tools
    tools = [
        ToolSpec(
            name="get_weather",
            description="Get weather by city",
            parameters_json_schema={"type":"object","properties":{"city":{"type":"string"}},"required":["city"]},
        )
    ]

    # Stream with tools
    async for ev in hub.stream(
        provider="openai",
        model="o3-mini",
        messages="Is it raining in Seattle? Use a tool if needed.",
        tools=tools,
        reasoning_effort="medium",
    ):
        if ev.type == "text.delta":
            print(ev.data, end="", flush=True)
        elif ev.type == "tool.call.delta":
            print(f"\nTool call: {ev.data}")
        elif ev.type == "usage":
            print(f"\nUsage: {ev.data}")

if __name__ == "__main__":
    asyncio.run(main())
