"""CPG JSON type definitions for greploom.

These types mirror treeloom's serialization format but carry no runtime
dependency on treeloom itself.  Node IDs are opaque strings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class NodeKind(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    PARAMETER = "parameter"
    VARIABLE = "variable"
    CALL = "call"
    LITERAL = "literal"
    RETURN = "return"
    IMPORT = "import"
    BRANCH = "branch"
    LOOP = "loop"
    BLOCK = "block"


class EdgeKind(str, Enum):
    CONTAINS = "contains"
    HAS_PARAMETER = "has_parameter"
    HAS_RETURN_TYPE = "has_return_type"
    FLOWS_TO = "flows_to"
    BRANCHES_TO = "branches_to"
    DATA_FLOWS_TO = "data_flows_to"
    DEFINED_BY = "defined_by"
    USED_BY = "used_by"
    CALLS = "calls"
    RESOLVES_TO = "resolves_to"
    IMPORTS = "imports"


@dataclass
class SourceLocation:
    file: str
    line: int
    column: int = 0


@dataclass
class CpgNode:
    id: str
    kind: NodeKind
    name: str
    location: SourceLocation | None = None
    scope: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class CpgEdge:
    source: str
    target: str
    kind: EdgeKind
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class CpgData:
    treeloom_version: str
    nodes: list[CpgNode]
    edges: list[CpgEdge]
    annotations: dict[str, dict[str, Any]] = field(default_factory=dict)
    edge_annotations: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Deserialization helpers
# ---------------------------------------------------------------------------


def _load_location(raw: dict[str, Any] | None) -> SourceLocation | None:
    if raw is None:
        return None
    return SourceLocation(
        file=raw["file"],
        line=raw["line"],
        column=raw.get("column", 0),
    )


def _load_node(raw: dict[str, Any]) -> CpgNode:
    return CpgNode(
        id=raw["id"],
        kind=NodeKind(raw["kind"]),
        name=raw["name"],
        location=_load_location(raw.get("location")),
        scope=raw.get("scope"),
        attrs=raw.get("attrs", {}),
    )


def _load_edge(raw: dict[str, Any]) -> CpgEdge:
    return CpgEdge(
        source=raw["source"],
        target=raw["target"],
        kind=EdgeKind(raw["kind"]),
        attrs=raw.get("attrs", {}),
    )


def load_cpg(path: Path) -> CpgData:
    """Read a treeloom CPG JSON file and return a ``CpgData`` object.

    Raises ``KeyError`` if required top-level keys are missing, ``ValueError``
    if an unknown node or edge kind is encountered, and ``json.JSONDecodeError``
    if the file is not valid JSON.
    """
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    return CpgData(
        treeloom_version=raw["treeloom_version"],
        nodes=[_load_node(n) for n in raw.get("nodes", [])],
        edges=[_load_edge(e) for e in raw.get("edges", [])],
        annotations=raw.get("annotations", {}),
        edge_annotations=raw.get("edge_annotations", []),
    )
