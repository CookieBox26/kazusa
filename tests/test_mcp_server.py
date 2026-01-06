import json
import pytest
import pytest_asyncio
import importlib.resources
from fastmcp.client import Client
from fastmcp.client.transports import FastMCPTransport
from kazusa.server import mcp, initialize


@pytest_asyncio.fixture
async def mcp_client():
    test_data_dir = importlib.resources.files('kazusa').parent / 'tests/data/'
    references_file = test_data_dir / "references.toml"
    index_file = test_data_dir / "faiss.index"
    initialize(references_file, index_file)
    async with Client(transport=FastMCPTransport(mcp)) as client:
        yield client


@pytest.mark.asyncio
async def test_list_tools(mcp_client: Client):
    tools = await mcp_client.list_tools()
    tool_names = [tool.name for tool in tools]
    assert "search_references" in tool_names
    assert "list_all_references" in tool_names
    assert "get_reference_by_arxiv" in tool_names


@pytest.mark.asyncio
async def test_search_references(mcp_client: Client):
    result = await mcp_client.call_tool(
        name="search_references",
        arguments={"query": "Transformer", "top_k": 3}
    )
    data = json.loads(result.content[0].text)
    assert isinstance(data, list)
    if len(data) > 0:
        assert "_similarity_score" in data[0]


@pytest.mark.asyncio
async def test_get_reference_by_arxiv(mcp_client: Client):
    result = await mcp_client.call_tool(
        name="get_reference_by_arxiv",
        arguments={"arxiv_id": "1706.03762"}
    )
    data = json.loads(result.content[0].text)
    if data is not None:
        assert data["title"] == "Attention Is All You Need"
