"""Tests for greploom.search.expand."""

from __future__ import annotations

import pytest

from greploom.cpg_types import CpgData, CpgEdge, CpgNode, EdgeKind, NodeKind
from greploom.search.expand import ExpandedNode, expand_hits

# ---------------------------------------------------------------------------
# Fixture CPG
#
#  mod1 (MODULE)
#  ├── cls1 (CLASS, scope=mod1)
#  │    └── fn1 (FUNCTION, scope=cls1)  ── CALLS ──► fn2
#  │         ├── param1 (PARAMETER)
#  │         └── param2 (PARAMETER)
#  ├── fn2 (FUNCTION, scope=mod1)
#  ├── imp1 (IMPORT) ◄── IMPORTS ── mod1
#  └── var1 (VARIABLE) ── DATA_FLOWS_TO ──► fn1
# ---------------------------------------------------------------------------


def _make_cpg() -> CpgData:
    nodes = [
        CpgNode(id="mod1", kind=NodeKind.MODULE, name="mymodule"),
        CpgNode(id="cls1", kind=NodeKind.CLASS, name="MyClass", scope="mod1"),
        CpgNode(id="fn1", kind=NodeKind.FUNCTION, name="fn_one", scope="cls1"),
        CpgNode(id="fn2", kind=NodeKind.FUNCTION, name="fn_two", scope="mod1"),
        CpgNode(id="param1", kind=NodeKind.PARAMETER, name="x"),
        CpgNode(id="param2", kind=NodeKind.PARAMETER, name="y"),
        CpgNode(id="imp1", kind=NodeKind.IMPORT, name="os"),
        CpgNode(id="var1", kind=NodeKind.VARIABLE, name="result"),
    ]
    edges = [
        CpgEdge(source="mod1", target="cls1", kind=EdgeKind.CONTAINS),
        CpgEdge(source="mod1", target="fn2", kind=EdgeKind.CONTAINS),
        CpgEdge(source="cls1", target="fn1", kind=EdgeKind.CONTAINS),
        CpgEdge(source="fn1", target="fn2", kind=EdgeKind.CALLS),
        CpgEdge(source="fn1", target="param1", kind=EdgeKind.HAS_PARAMETER),
        CpgEdge(source="fn1", target="param2", kind=EdgeKind.HAS_PARAMETER),
        CpgEdge(source="mod1", target="imp1", kind=EdgeKind.IMPORTS),
        CpgEdge(source="var1", target="fn1", kind=EdgeKind.DATA_FLOWS_TO),
    ]
    return CpgData(treeloom_version="0.3.0", nodes=nodes, edges=edges)


@pytest.fixture()
def cpg() -> CpgData:
    return _make_cpg()


def _ids(nodes: list[ExpandedNode]) -> set[str]:
    return {e.node.id for e in nodes}


def _by_id(nodes: list[ExpandedNode], node_id: str) -> ExpandedNode:
    return next(e for e in nodes if e.node.id == node_id)


# ---------------------------------------------------------------------------
# Core expansion from fn1
# ---------------------------------------------------------------------------


def test_expand_fn1_includes_hit(cpg):
    result = expand_hits(["fn1"], cpg)
    hit = _by_id(result, "fn1")
    assert hit.relationship == "hit"
    assert hit.relevance == 1.0


def test_expand_fn1_includes_callee_fn2(cpg):
    result = expand_hits(["fn1"], cpg)
    assert "fn2" in _ids(result)
    assert _by_id(result, "fn2").relationship == "callee"


def test_expand_fn1_includes_parameters(cpg):
    result = expand_hits(["fn1"], cpg)
    assert {"param1", "param2"}.issubset(_ids(result))
    for pid in ("param1", "param2"):
        assert _by_id(result, pid).relationship == "parameter"


def test_expand_fn1_includes_class_parent(cpg):
    result = expand_hits(["fn1"], cpg)
    assert "cls1" in _ids(result)
    assert _by_id(result, "cls1").relationship == "class"


def test_expand_fn1_includes_data_source(cpg):
    result = expand_hits(["fn1"], cpg)
    assert "var1" in _ids(result)
    assert _by_id(result, "var1").relationship == "data_source"


def test_expand_fn1_includes_module_import(cpg):
    result = expand_hits(["fn1"], cpg)
    assert "imp1" in _ids(result)
    assert _by_id(result, "imp1").relationship == "import"


# ---------------------------------------------------------------------------
# Expansion from fn2 — caller direction
# ---------------------------------------------------------------------------


def test_expand_fn2_includes_caller_fn1(cpg):
    result = expand_hits(["fn2"], cpg)
    assert "fn1" in _ids(result)
    assert _by_id(result, "fn1").relationship == "caller"


# ---------------------------------------------------------------------------
# Dedup: hit wins over callee
# ---------------------------------------------------------------------------


def test_dedup_hit_wins_over_callee(cpg):
    # fn2 is a callee of fn1, but also a direct hit — hit relevance (1.0) wins.
    result = expand_hits(["fn1", "fn2"], cpg)
    fn2 = _by_id(result, "fn2")
    assert fn2.relevance == 1.0
    assert fn2.relationship == "hit"


# ---------------------------------------------------------------------------
# Unknown node ID → empty result
# ---------------------------------------------------------------------------


def test_unknown_node_id_returns_empty(cpg):
    result = expand_hits(["does_not_exist"], cpg)
    assert result == []


# ---------------------------------------------------------------------------
# depth=2: callers-of-callers
# ---------------------------------------------------------------------------


def test_depth2_expands_callers_of_callees(cpg):
    """With depth=2, expanding from fn2 should reach fn1's parameters (via fn1 as caller)."""
    result = expand_hits(["fn2"], cpg, depth=2)
    ids = _ids(result)
    # fn1 is a caller of fn2; at depth 2 fn1's parameters/class should appear
    assert "fn1" in ids
    assert "param1" in ids or "param2" in ids  # depth-2 neighbors of fn1


# ---------------------------------------------------------------------------
# Sort order: relevance descending
# ---------------------------------------------------------------------------


def test_result_sorted_by_relevance_descending(cpg):
    result = expand_hits(["fn1"], cpg)
    relevances = [e.relevance for e in result]
    assert relevances == sorted(relevances, reverse=True)


# ---------------------------------------------------------------------------
# Relevance decay values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "node_id,expected_rel",
    [
        ("fn2", 0.8),    # callee
        ("cls1", 0.7),   # class
        ("param1", 0.6), # parameter
        ("var1", 0.5),   # data_source
        ("imp1", 0.3),   # import
    ],
)
def test_relevance_decay(cpg, node_id, expected_rel):
    result = expand_hits(["fn1"], cpg)
    node = _by_id(result, node_id)
    assert abs(node.relevance - expected_rel) < 1e-9
