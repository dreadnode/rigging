"""Search the web through Rigging's MCP tools without model or Parallel API keys.

Run after installing Rigging from this checkout:
    python examples/parallel_search.py "Rigging MCP documentation"
"""

import argparse
import asyncio

import rigging as rg


async def search(query: str) -> None:
    async with rg.mcp(
        "streamable-http",
        url="https://search.parallel.ai/mcp",
        headers={"User-Agent": f"rigging/{rg.__version__}"},
    ) as client:
        tool = next(tool for tool in client.tools if tool.name == "web_search")
        parts = await tool(objective=query, search_queries=[query])
        for part in parts:
            print(part.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Web search query")
    asyncio.run(search(parser.parse_args().query))
