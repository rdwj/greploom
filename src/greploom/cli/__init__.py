"""greploom CLI — semantic code search with graph-aware context retrieval."""

from __future__ import annotations

import click

from greploom.version import __version__

from .index_cmd import index
from .query_cmd import query
from .serve_cmd import serve


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Semantic code search with graph-aware context retrieval."""


main.add_command(index)
main.add_command(query)
main.add_command(serve)
