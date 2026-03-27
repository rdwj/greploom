"""greploom serve command — start the MCP server."""

from __future__ import annotations

import click

from greploom.config import GrepLoomConfig


@click.command()
@click.option("--db", "db_path", default=None, help="SQLite database path.")
@click.option(
    "--cpg",
    "cpg_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Default CPG JSON path.",
)
@click.option("--host", default="0.0.0.0", show_default=True, help="Host to bind.")
@click.option("--port", default=8901, show_default=True, type=int, help="Port to listen on.")
@click.option(
    "--transport",
    default="streamable-http",
    show_default=True,
    type=click.Choice(["streamable-http", "stdio"]),
    help="MCP transport.",
)
def serve(db_path: str | None, cpg_path: str | None, host: str, port: int, transport: str) -> None:
    """Start the MCP server."""
    from greploom.mcp.server import create_server  # deferred — fastmcp is optional

    config = GrepLoomConfig.from_env()
    if db_path:
        config = GrepLoomConfig(
            embedding_url=config.embedding_url,
            embedding_model=config.embedding_model,
            db_path=db_path,
            token_budget=config.token_budget,
            summary_tier=config.summary_tier,
        )

    server = create_server(config)
    server.run(transport=transport, host=host, port=port)
