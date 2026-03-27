"""Tests for the greploom MCP server factory and tool registration."""

from __future__ import annotations

import asyncio

import pytest
from fastmcp import FastMCP

from greploom.mcp.server import create_server


def test_create_server_returns_fastmcp_instance() -> None:
    server = create_server()
    assert isinstance(server, FastMCP)


def test_create_server_with_config() -> None:
    from greploom.config import GrepLoomConfig

    config = GrepLoomConfig(db_path=".greploom/test.db")
    server = create_server(config)
    assert isinstance(server, FastMCP)


@pytest.fixture()
def server() -> FastMCP:
    return create_server()


def test_search_code_tool_registered(server: FastMCP) -> None:
    tools = asyncio.run(server.list_tools())
    names = [t.name for t in tools]
    assert "search_code" in names, f"Expected 'search_code' in tools, got: {names}"


def test_index_code_tool_registered(server: FastMCP) -> None:
    tools = asyncio.run(server.list_tools())
    names = [t.name for t in tools]
    assert "index_code" in names, f"Expected 'index_code' in tools, got: {names}"


def test_server_has_exactly_two_tools(server: FastMCP) -> None:
    tools = asyncio.run(server.list_tools())
    assert len(tools) == 2, f"Expected 2 tools, got {len(tools)}: {[t.name for t in tools]}"
