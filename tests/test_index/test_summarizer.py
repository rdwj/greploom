"""Tests for greploom.index.summarizer."""

from __future__ import annotations

import pytest

from greploom.cpg_types import CpgData, CpgEdge, CpgNode, EdgeKind, NodeKind, SourceLocation
from greploom.index.summarizer import (
    INDEXABLE_KINDS,
    build_edges_from,
    build_node_lookup,
    summarize_node,
)

# ---------------------------------------------------------------------------
# Fixture: a small CPG with one module, one class with two methods,
# and one standalone function that calls a helper.
# ---------------------------------------------------------------------------
#
#  module: src/app.py
#    ├── class Widget (line 5)
#    │     ├── method __init__(self) (line 6)  [no callees]
#    │     └── method render(self, ctx) (line 10)  calls: helper
#    └── function standalone(x, y) (line 20)  calls: helper
#                                              decorators: ["app.route"]
#
#  module: src/util.py
#    └── function helper() (line 1)
#

_LOC = {
    "mod_app": SourceLocation(file="src/app.py", line=1),
    "cls_widget": SourceLocation(file="src/app.py", line=5),
    "fn_init": SourceLocation(file="src/app.py", line=6),
    "fn_render": SourceLocation(file="src/app.py", line=10),
    "fn_standalone": SourceLocation(file="src/app.py", line=20),
    "mod_util": SourceLocation(file="src/util.py", line=1),
    "fn_helper": SourceLocation(file="src/util.py", line=1),
}

_NODES = [
    CpgNode(id="mod:app", kind=NodeKind.MODULE, name="app", location=_LOC["mod_app"]),
    CpgNode(id="cls:widget", kind=NodeKind.CLASS, name="Widget", location=_LOC["cls_widget"]),
    CpgNode(id="fn:init", kind=NodeKind.FUNCTION, name="__init__", location=_LOC["fn_init"]),
    CpgNode(
        id="fn:render",
        kind=NodeKind.FUNCTION,
        name="render",
        location=_LOC["fn_render"],
    ),
    CpgNode(
        id="fn:standalone",
        kind=NodeKind.FUNCTION,
        name="standalone",
        location=_LOC["fn_standalone"],
        attrs={"decorators": ["app.route"]},
    ),
    CpgNode(id="mod:util", kind=NodeKind.MODULE, name="util", location=_LOC["mod_util"]),
    CpgNode(id="fn:helper", kind=NodeKind.FUNCTION, name="helper", location=_LOC["fn_helper"]),
    # Parameter nodes
    CpgNode(id="param:self1", kind=NodeKind.PARAMETER, name="self"),
    CpgNode(id="param:self2", kind=NodeKind.PARAMETER, name="self"),
    CpgNode(id="param:ctx", kind=NodeKind.PARAMETER, name="ctx"),
    CpgNode(id="param:x", kind=NodeKind.PARAMETER, name="x"),
    CpgNode(id="param:y", kind=NodeKind.PARAMETER, name="y"),
    # A VARIABLE node — must not be indexed
    CpgNode(id="var:count", kind=NodeKind.VARIABLE, name="count"),
]

_EDGES = [
    # Module → class/function membership
    CpgEdge(source="mod:app", target="cls:widget", kind=EdgeKind.CONTAINS),
    CpgEdge(source="mod:app", target="fn:standalone", kind=EdgeKind.CONTAINS),
    CpgEdge(source="mod:util", target="fn:helper", kind=EdgeKind.CONTAINS),
    # Class → method membership
    CpgEdge(source="cls:widget", target="fn:init", kind=EdgeKind.CONTAINS),
    CpgEdge(source="cls:widget", target="fn:render", kind=EdgeKind.CONTAINS),
    # Parameters
    CpgEdge(source="fn:init", target="param:self1", kind=EdgeKind.HAS_PARAMETER),
    CpgEdge(source="fn:render", target="param:self2", kind=EdgeKind.HAS_PARAMETER),
    CpgEdge(source="fn:render", target="param:ctx", kind=EdgeKind.HAS_PARAMETER),
    CpgEdge(source="fn:standalone", target="param:x", kind=EdgeKind.HAS_PARAMETER),
    CpgEdge(source="fn:standalone", target="param:y", kind=EdgeKind.HAS_PARAMETER),
    # Calls
    CpgEdge(source="fn:render", target="fn:helper", kind=EdgeKind.CALLS),
    CpgEdge(source="fn:standalone", target="fn:helper", kind=EdgeKind.CALLS),
]


@pytest.fixture()
def cpg() -> CpgData:
    return CpgData(treeloom_version="0.3.0", nodes=_NODES, edges=_EDGES)


@pytest.fixture()
def node_lookup(cpg):
    return build_node_lookup(cpg)


@pytest.fixture()
def edges_from(cpg):
    return build_edges_from(cpg)


# ---------------------------------------------------------------------------
# build_node_lookup / build_edges_from
# ---------------------------------------------------------------------------


def test_build_node_lookup_all_ids(cpg, node_lookup):
    assert set(node_lookup.keys()) == {n.id for n in cpg.nodes}


def test_build_node_lookup_retrieves_correct_node(node_lookup):
    node = node_lookup["cls:widget"]
    assert node.kind is NodeKind.CLASS
    assert node.name == "Widget"


def test_build_edges_from_outgoing_count(edges_from):
    # fn:render has: HAS_PARAMETER×2 + CALLS×1
    assert len(edges_from["fn:render"]) == 3


def test_build_edges_from_missing_node_returns_empty(edges_from):
    assert edges_from.get("nonexistent", []) == []


# ---------------------------------------------------------------------------
# INDEXABLE_KINDS constant
# ---------------------------------------------------------------------------


def test_indexable_kinds_contains_expected():
    assert INDEXABLE_KINDS == {NodeKind.FUNCTION, NodeKind.CLASS, NodeKind.MODULE}


# ---------------------------------------------------------------------------
# Non-indexable kinds return None
# ---------------------------------------------------------------------------

NON_INDEXABLE = [
    NodeKind.PARAMETER,
    NodeKind.VARIABLE,
    NodeKind.CALL,
    NodeKind.LITERAL,
    NodeKind.RETURN,
    NodeKind.IMPORT,
    NodeKind.BRANCH,
    NodeKind.LOOP,
    NodeKind.BLOCK,
]


@pytest.mark.parametrize("kind", NON_INDEXABLE)
def test_non_indexable_kinds_return_none(kind, node_lookup, edges_from):
    node = CpgNode(id="tmp", kind=kind, name="x")
    assert summarize_node(node, node_lookup, edges_from) is None
    assert summarize_node(node, node_lookup, edges_from, tier="fast") is None


# ---------------------------------------------------------------------------
# Fast tier — signature format
# ---------------------------------------------------------------------------

FAST_TIER_CASES = [
    # (node_id, expected_substring_or_exact)
    ("fn:init", "function __init__(self) in src/app.py:6"),
    ("fn:render", "function render(self, ctx) in src/app.py:10"),
    ("fn:standalone", "function standalone(x, y) in src/app.py:20"),
    ("cls:widget", "class Widget in src/app.py:5"),
    ("mod:app", "module src/app.py"),
    ("mod:util", "module src/util.py"),
]


@pytest.mark.parametrize("node_id,expected", FAST_TIER_CASES)
def test_fast_tier(node_id, expected, node_lookup, edges_from):
    node = node_lookup[node_id]
    result = summarize_node(node, node_lookup, edges_from, tier="fast")
    assert result == expected, f"node={node_id!r}: {result!r} != {expected!r}"


def test_fast_tier_no_location(node_lookup, edges_from):
    node = CpgNode(id="fn:noloc", kind=NodeKind.FUNCTION, name="anon")
    result = summarize_node(node, node_lookup, edges_from, tier="fast")
    assert result == "function anon() in <unknown>"


# ---------------------------------------------------------------------------
# Enhanced tier — structural context
# ---------------------------------------------------------------------------


def test_enhanced_function_with_params_and_callee(node_lookup, edges_from):
    result = summarize_node(node_lookup["fn:render"], node_lookup, edges_from)
    assert result is not None
    lines = result.splitlines()
    assert lines[0] == "function render(self, ctx) in src/app.py:10"
    assert any("Parameters:" in ln for ln in lines)
    assert any("Calls:" in ln and "helper" in ln for ln in lines)


def test_enhanced_function_with_decorators(node_lookup, edges_from):
    result = summarize_node(node_lookup["fn:standalone"], node_lookup, edges_from)
    assert result is not None
    assert "Decorators: app.route" in result


def test_enhanced_function_no_extra_lines_when_empty(node_lookup, edges_from):
    """fn:helper has no parameters, no callees, no decorators."""
    result = summarize_node(node_lookup["fn:helper"], node_lookup, edges_from)
    assert result is not None
    # Should be just a single line — no Parameters/Calls/Decorators sections
    assert "Parameters:" not in result
    assert "Calls:" not in result
    assert "Decorators:" not in result


def test_enhanced_class_includes_methods(node_lookup, edges_from):
    result = summarize_node(node_lookup["cls:widget"], node_lookup, edges_from)
    assert result is not None
    assert "Methods:" in result
    assert "__init__" in result
    assert "render" in result


def test_enhanced_class_with_decorators(node_lookup, edges_from):
    node = CpgNode(
        id="cls:dec",
        kind=NodeKind.CLASS,
        name="Decorated",
        location=SourceLocation(file="src/x.py", line=1),
        attrs={"decorators": ["dataclass", "frozen"]},
    )
    result = summarize_node(node, node_lookup, edges_from)
    assert result is not None
    assert "Decorators: dataclass, frozen" in result


def test_enhanced_module_includes_top_level_members(node_lookup, edges_from):
    result = summarize_node(node_lookup["mod:app"], node_lookup, edges_from)
    assert result is not None
    assert "Contains:" in result
    assert "Widget" in result
    assert "standalone" in result


def test_enhanced_module_no_contains_when_empty(node_lookup, edges_from):
    """A module with no CONTAINS edges should not emit a Contains line."""
    node = CpgNode(
        id="mod:empty",
        kind=NodeKind.MODULE,
        name="empty",
        location=SourceLocation(file="src/empty.py", line=1),
    )
    result = summarize_node(node, node_lookup, edges_from)
    assert result is not None
    assert "Contains:" not in result


def test_enhanced_unknown_tier_falls_back_to_fast(node_lookup, edges_from):
    """An unrecognised tier name should fall back to fast-tier output."""
    fast = summarize_node(node_lookup["fn:helper"], node_lookup, edges_from, tier="fast")
    unknown = summarize_node(node_lookup["fn:helper"], node_lookup, edges_from, tier="turbo")
    # fast tier is a single line; unknown tier must also be a single line
    assert fast == unknown


# ---------------------------------------------------------------------------
# Graceful handling of missing / empty attrs
# ---------------------------------------------------------------------------


def test_no_attrs_key_does_not_raise(node_lookup, edges_from):
    node = CpgNode(
        id="fn:bare",
        kind=NodeKind.FUNCTION,
        name="bare",
        location=SourceLocation(file="src/bare.py", line=1),
    )
    result = summarize_node(node, node_lookup, edges_from)
    assert result is not None
    assert result.startswith("function bare()")


def test_empty_decorators_list_not_emitted(node_lookup, edges_from):
    node = CpgNode(
        id="fn:nodec",
        kind=NodeKind.FUNCTION,
        name="nodec",
        location=SourceLocation(file="src/x.py", line=1),
        attrs={"decorators": []},
    )
    result = summarize_node(node, node_lookup, edges_from)
    assert result is not None
    assert "Decorators:" not in result


def test_callee_not_in_lookup_is_skipped(edges_from):
    """CALLS edge pointing to an unknown node ID must not crash."""
    orphan_node = CpgNode(id="fn:orphan", kind=NodeKind.FUNCTION, name="orphan")
    orphan_lookup = {"fn:orphan": orphan_node}
    orphan_edges = {
        "fn:orphan": [CpgEdge(source="fn:orphan", target="fn:GONE", kind=EdgeKind.CALLS)]
    }
    result = summarize_node(orphan_node, orphan_lookup, orphan_edges)
    assert result is not None
    assert "Calls:" not in result
