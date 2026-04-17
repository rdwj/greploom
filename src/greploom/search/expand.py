"""Graph expansion: walk CPG edges from search hits to assemble context neighborhoods."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TypedDict

from greploom.cpg_types import CpgData, CpgEdge, CpgNode, EdgeKind, NodeKind
from greploom.index.summarizer import build_edges_from, build_node_lookup


class NodeRef(TypedDict):
    node_id: str
    name: str
    file: str | None
    line: int | None


class StructuralContext(TypedDict):
    callers: list[NodeRef]
    callees: list[NodeRef]
    parameters: list[NodeRef]
    parent_class: NodeRef | None
    data_sources: list[NodeRef]
    imports: list[NodeRef]


# Relationship priority for stable sort (lower = higher priority).
_REL_PRIORITY = {
    "hit": 0,
    "caller": 1,
    "callee": 2,
    "class": 3,
    "parameter": 4,
    "data_source": 5,
    "import": 6,
}


@dataclass
class ExpandedNode:
    node: CpgNode
    relevance: float  # 1.0 for hit, decaying by relationship
    relationship: str  # "hit", "caller", "callee", "parameter", "class", "import", "data_source"
    structural_context: StructuralContext | None = None


def _node_ref(node: CpgNode) -> NodeRef:
    return NodeRef(
        node_id=node.id,
        name=node.name,
        file=node.location.file if node.location else None,
        line=node.location.line if node.location else None,
    )


def _build_edges_to(cpg: CpgData) -> dict[str, list[CpgEdge]]:
    """Return mapping of target node ID → incoming edges."""
    result: dict[str, list[CpgEdge]] = defaultdict(list)
    for edge in cpg.edges:
        result[edge.target].append(edge)
    return dict(result)


def _module_ancestor(node_id: str, node_lookup: dict[str, CpgNode]) -> CpgNode | None:
    """Walk the scope chain upward until a MODULE node is found."""
    visited: set[str] = set()
    current_id: str | None = node_id
    while current_id and current_id not in visited:
        visited.add(current_id)
        node = node_lookup.get(current_id)
        if node is None:
            return None
        if node.kind is NodeKind.MODULE:
            return node
        current_id = node.scope
    return None


def _expand_one(
    node_id: str,
    relevance: float,
    node_lookup: dict[str, CpgNode],
    edges_from: dict[str, list[CpgEdge]],
    edges_to: dict[str, list[CpgEdge]],
    *,
    include_imports: bool,
) -> list[tuple[str, float, str]]:
    """Return (node_id, relevance, relationship) tuples for neighbors of *node_id*."""
    results: list[tuple[str, float, str]] = []

    outgoing = edges_from.get(node_id, [])
    incoming = edges_to.get(node_id, [])

    # Callees: forward CALLS
    for edge in outgoing:
        if edge.kind is EdgeKind.CALLS and edge.target in node_lookup:
            results.append((edge.target, relevance * 0.8, "callee"))

    # Callers: backward CALLS
    for edge in incoming:
        if edge.kind is EdgeKind.CALLS and edge.source in node_lookup:
            results.append((edge.source, relevance * 0.8, "caller"))

    # Parameters: forward HAS_PARAMETER
    for edge in outgoing:
        if edge.kind is EdgeKind.HAS_PARAMETER and edge.target in node_lookup:
            results.append((edge.target, relevance * 0.6, "parameter"))

    # Class parent: scope points to a CLASS node
    node = node_lookup.get(node_id)
    if node and node.scope:
        parent = node_lookup.get(node.scope)
        if parent and parent.kind is NodeKind.CLASS:
            results.append((parent.id, relevance * 0.7, "class"))

    # Data flow sources: backward DATA_FLOWS_TO
    for edge in incoming:
        if edge.kind is EdgeKind.DATA_FLOWS_TO and edge.source in node_lookup:
            results.append((edge.source, relevance * 0.5, "data_source"))

    # Imports: forward IMPORTS from the node's MODULE ancestor
    if include_imports:
        module = _module_ancestor(node_id, node_lookup)
        if module:
            for edge in edges_from.get(module.id, []):
                if edge.kind is EdgeKind.IMPORTS and edge.target in node_lookup:
                    results.append((edge.target, relevance * 0.3, "import"))

    return results


def _build_structural_context(
    node_id: str,
    node_lookup: dict[str, CpgNode],
    edges_from: dict[str, list[CpgEdge]],
    edges_to: dict[str, list[CpgEdge]],
) -> StructuralContext:
    """Return structured relationship data for a single CPG node."""
    outgoing = edges_from.get(node_id, [])
    incoming = edges_to.get(node_id, [])

    callers = [
        _node_ref(node_lookup[e.source])
        for e in incoming
        if e.kind is EdgeKind.CALLS and e.source in node_lookup
    ]
    callees = [
        _node_ref(node_lookup[e.target])
        for e in outgoing
        if e.kind is EdgeKind.CALLS and e.target in node_lookup
    ]
    parameters = [
        _node_ref(node_lookup[e.target])
        for e in outgoing
        if e.kind is EdgeKind.HAS_PARAMETER and e.target in node_lookup
    ]

    parent_class: NodeRef | None = None
    node = node_lookup.get(node_id)
    if node and node.scope:
        parent = node_lookup.get(node.scope)
        if parent and parent.kind is NodeKind.CLASS:
            parent_class = _node_ref(parent)

    data_sources = [
        _node_ref(node_lookup[e.source])
        for e in incoming
        if e.kind is EdgeKind.DATA_FLOWS_TO and e.source in node_lookup
    ]

    imports: list[NodeRef] = []
    module = _module_ancestor(node_id, node_lookup)
    if module:
        imports = [
            _node_ref(node_lookup[e.target])
            for e in edges_from.get(module.id, [])
            if e.kind is EdgeKind.IMPORTS and e.target in node_lookup
        ]

    return StructuralContext(
        callers=callers,
        callees=callees,
        parameters=parameters,
        parent_class=parent_class,
        data_sources=data_sources,
        imports=imports,
    )


def expand_hits(
    hit_node_ids: list[str],
    cpg: CpgData,
    depth: int = 1,
) -> list[ExpandedNode]:
    """Walk the CPG from *hit_node_ids* and return context neighborhoods.

    Nodes appearing via multiple paths keep the highest relevance seen.
    Results are sorted by relevance descending, then relationship priority.
    """
    node_lookup = build_node_lookup(cpg)
    edges_from = build_edges_from(cpg)
    edges_to = _build_edges_to(cpg)

    # best[(node_id)] = (relevance, relationship)
    best: dict[str, tuple[float, str]] = {}

    def _record(node_id: str, rel: float, relationship: str) -> None:
        prev = best.get(node_id)
        if prev is None or rel > prev[0]:
            best[node_id] = (rel, relationship)

    # Seed with hits
    valid_hits = [nid for nid in hit_node_ids if nid in node_lookup]
    for nid in valid_hits:
        _record(nid, 1.0, "hit")

    # BFS expansion, import edges are non-recursive
    frontier = list(valid_hits)
    for _ in range(depth):
        next_frontier: list[str] = []
        for nid in frontier:
            current_rel = best[nid][0]
            neighbors = _expand_one(
                nid, current_rel, node_lookup, edges_from, edges_to, include_imports=True
            )
            for target_id, rel, relationship in neighbors:
                if relationship == "import":
                    _record(target_id, rel, relationship)
                    # imports do not recurse — do not add to next_frontier
                else:
                    old = best.get(target_id)
                    _record(target_id, rel, relationship)
                    # Only recurse if this is a new node or improved relevance
                    if old is None or rel > old[0]:
                        next_frontier.append(target_id)
        frontier = next_frontier

    expanded = [
        ExpandedNode(
            node=node_lookup[nid],
            relevance=rel,
            relationship=relationship,
            structural_context=_build_structural_context(nid, node_lookup, edges_from, edges_to),
        )
        for nid, (rel, relationship) in best.items()
    ]
    expanded.sort(key=lambda e: (-e.relevance, _REL_PRIORITY.get(e.relationship, 99)))
    return expanded
