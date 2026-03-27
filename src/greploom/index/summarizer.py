"""Generates plain-text summaries of CPG nodes for embedding.

Only FUNCTION, CLASS, and MODULE nodes are indexed; all other kinds return None.
Tiers: ``fast`` (signature only) and ``enhanced`` (adds params, callees, members).
"""

from __future__ import annotations

from collections import defaultdict

from greploom.cpg_types import CpgData, CpgEdge, CpgNode, EdgeKind, NodeKind

INDEXABLE_KINDS = frozenset({NodeKind.FUNCTION, NodeKind.CLASS, NodeKind.MODULE})


def build_node_lookup(cpg: CpgData) -> dict[str, CpgNode]:
    """Return a mapping of node ID → CpgNode for fast resolution."""
    return {node.id: node for node in cpg.nodes}


def build_edges_from(cpg: CpgData) -> dict[str, list[CpgEdge]]:
    """Return a mapping of source node ID → outgoing edges."""
    result: dict[str, list[CpgEdge]] = defaultdict(list)
    for edge in cpg.edges:
        result[edge.source].append(edge)
    return dict(result)



def _file_line(node: CpgNode) -> str:
    return f"{node.location.file}:{node.location.line}" if node.location else "<unknown>"


def _related_names(
    node_id: str,
    edge_kind: EdgeKind,
    node_lookup: dict[str, CpgNode],
    edges_from: dict[str, list[CpgEdge]],
    *,
    child_kinds: frozenset[NodeKind] | None = None,
) -> list[str]:
    """Names of targets reachable via *edge_kind*; filtered by *child_kinds* when given."""
    names = []
    for edge in edges_from.get(node_id, []):
        if edge.kind != edge_kind:
            continue
        target = node_lookup.get(edge.target)
        if target is None:
            continue
        if child_kinds is None or target.kind in child_kinds:
            names.append(target.name)
    return names



def summarize_node(
    node: CpgNode,
    node_lookup: dict[str, CpgNode],
    edges_from: dict[str, list[CpgEdge]],
    tier: str = "enhanced",
) -> str | None:
    """Return a plain-text summary of *node* suitable for embedding.

    Returns ``None`` for non-indexable node kinds.  Unknown *tier* values fall
    back to ``fast``.
    """
    if node.kind not in INDEXABLE_KINDS:
        return None

    loc = _file_line(node)
    params = _related_names(node.id, EdgeKind.HAS_PARAMETER, node_lookup, edges_from)

    # -- fast tier -----------------------------------------------------------
    if node.kind == NodeKind.FUNCTION:
        signature = f"function {node.name}({', '.join(params)}) in {loc}"
    elif node.kind == NodeKind.CLASS:
        signature = f"class {node.name} in {loc}"
    else:  # MODULE
        signature = f"module {node.location.file if node.location else loc}"

    if tier == "fast":
        return signature

    # -- enhanced tier -------------------------------------------------------
    parts = [signature]

    if node.kind == NodeKind.FUNCTION:
        if params:
            parts.append(f"Parameters: {', '.join(params)}")
        callees = _related_names(node.id, EdgeKind.CALLS, node_lookup, edges_from)
        if callees:
            parts.append(f"Calls: {', '.join(callees)}")
        decorators: list[str] = node.attrs.get("decorators", [])
        if decorators:
            parts.append(f"Decorators: {', '.join(decorators)}")

    elif node.kind == NodeKind.CLASS:
        methods = _related_names(
            node.id, EdgeKind.CONTAINS, node_lookup, edges_from,
            child_kinds=frozenset({NodeKind.FUNCTION}),
        )
        if methods:
            parts.append(f"Methods: {', '.join(methods)}")
        decorators = node.attrs.get("decorators", [])
        if decorators:
            parts.append(f"Decorators: {', '.join(decorators)}")

    else:  # MODULE
        members = _related_names(
            node.id, EdgeKind.CONTAINS, node_lookup, edges_from,
            child_kinds=frozenset({NodeKind.FUNCTION, NodeKind.CLASS}),
        )
        if members:
            parts.append(f"Contains: {', '.join(members)}")

    return "\n".join(parts)
