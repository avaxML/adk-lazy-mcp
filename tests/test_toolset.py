import asyncio

from adk_lazy_mcp import LazyMCPToolset, RegistryConfig, ServerConfig


def test_discover_inspect_execute_happy_path():
    async def list_tools(server: str):
        return [
            {
                "name": "read_file",
                "description": "Read file text",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ]

    async def execute_tool(server: str, tool: str, arguments: dict):
        return {"isError": False, "content": [{"type": "text", "text": f"{arguments['path']}"}]}

    async def scenario():
        toolset = LazyMCPToolset(
            [ServerConfig(name="filesystem")],
            RegistryConfig(warm_mode="eager"),
            list_tools=list_tools,
            execute_tool=execute_tool,
        )
        await toolset.get_tools()
        discover = await toolset.discover_mcp_tools(query="read")
        assert discover["status"] == "success"
        assert discover["total_matches"] == 1

        inspect = await toolset.inspect_mcp_tool("filesystem", "read_file")
        assert inspect["required_fields"] == ["path"]

        run = await toolset.execute_mcp_tool("filesystem", "read_file", {"path": "README.md"})
        assert run["status"] == "success"
        assert run["tool_result"]["is_error"] is False

    asyncio.run(scenario())


def test_validation_error_is_structured():
    async def list_tools(server: str):
        return [
            {
                "name": "read_file",
                "description": "Read file text",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ]

    async def execute_tool(server: str, tool: str, arguments: dict):
        return {"isError": False, "content": []}

    async def scenario():
        toolset = LazyMCPToolset(
            [ServerConfig(name="filesystem")],
            RegistryConfig(warm_mode="eager"),
            list_tools=list_tools,
            execute_tool=execute_tool,
        )

        await toolset.get_tools()
        run = await toolset.execute_mcp_tool("filesystem", "read_file", {})
        assert run["status"] == "error"
        assert run["error_type"] == "validation_error"

    asyncio.run(scenario())
