"""greploom CLI — semantic code search with graph-aware context retrieval."""

import click

from greploom.version import __version__


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Semantic code search with graph-aware context retrieval."""


if __name__ == "__main__":
    main()
