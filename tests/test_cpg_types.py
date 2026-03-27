"""Tests for greploom.cpg_types."""

from __future__ import annotations

import json

import pytest

from greploom.cpg_types import (
    CpgData,
    CpgEdge,
    CpgNode,
    EdgeKind,
    NodeKind,
    SourceLocation,
    load_cpg,
)

# ---------------------------------------------------------------------------
# Enum coverage
# ---------------------------------------------------------------------------

NODE_KIND_VALUES = [
    ("module", NodeKind.MODULE),
    ("class", NodeKind.CLASS),
    ("function", NodeKind.FUNCTION),
    ("parameter", NodeKind.PARAMETER),
    ("variable", NodeKind.VARIABLE),
    ("call", NodeKind.CALL),
    ("literal", NodeKind.LITERAL),
    ("return", NodeKind.RETURN),
    ("import", NodeKind.IMPORT),
    ("branch", NodeKind.BRANCH),
    ("loop", NodeKind.LOOP),
    ("block", NodeKind.BLOCK),
]

EDGE_KIND_VALUES = [
    ("contains", EdgeKind.CONTAINS),
    ("has_parameter", EdgeKind.HAS_PARAMETER),
    ("has_return_type", EdgeKind.HAS_RETURN_TYPE),
    ("flows_to", EdgeKind.FLOWS_TO),
    ("branches_to", EdgeKind.BRANCHES_TO),
    ("data_flows_to", EdgeKind.DATA_FLOWS_TO),
    ("defined_by", EdgeKind.DEFINED_BY),
    ("used_by", EdgeKind.USED_BY),
    ("calls", EdgeKind.CALLS),
    ("resolves_to", EdgeKind.RESOLVES_TO),
    ("imports", EdgeKind.IMPORTS),
]


@pytest.mark.parametrize("value,expected", NODE_KIND_VALUES)
def test_node_kind_from_string(value, expected):
    assert NodeKind(value) is expected
    # str subclass — the enum value should compare equal to the raw string
    assert expected == value


@pytest.mark.parametrize("value,expected", EDGE_KIND_VALUES)
def test_edge_kind_from_string(value, expected):
    assert EdgeKind(value) is expected
    assert expected == value


def test_node_kind_invalid():
    with pytest.raises(ValueError, match="'unknown_node'"):
        NodeKind("unknown_node")


def test_edge_kind_invalid():
    with pytest.raises(ValueError, match="'unknown_edge'"):
        EdgeKind("unknown_edge")


# ---------------------------------------------------------------------------
# Dataclass construction
# ---------------------------------------------------------------------------

def test_source_location_defaults():
    loc = SourceLocation(file="src/foo.py", line=5)
    assert loc.column == 0


def test_source_location_explicit():
    loc = SourceLocation(file="src/bar.py", line=10, column=4)
    assert loc.file == "src/bar.py"
    assert loc.line == 10
    assert loc.column == 4


def test_cpg_node_minimal():
    node = CpgNode(id="function:src/foo.py:10:0:1", kind=NodeKind.FUNCTION, name="do_thing")
    assert node.location is None
    assert node.scope is None
    assert node.attrs == {}


def test_cpg_node_full():
    loc = SourceLocation(file="src/foo.py", line=10)
    node = CpgNode(
        id="function:src/foo.py:10:0:1",
        kind=NodeKind.FUNCTION,
        name="do_thing",
        location=loc,
        scope="module:src/foo.py:1:0:0",
        attrs={"is_async": False, "decorators": ["app.route"]},
    )
    assert node.kind is NodeKind.FUNCTION
    assert node.location.line == 10
    assert node.attrs["is_async"] is False


def test_cpg_edge_minimal():
    edge = CpgEdge(
        source="function:src/foo.py:10:0:1",
        target="call:src/foo.py:15:4:2",
        kind=EdgeKind.CALLS,
    )
    assert edge.attrs == {}


def test_cpg_edge_with_attrs():
    edge = CpgEdge(
        source="a", target="b", kind=EdgeKind.DATA_FLOWS_TO, attrs={"weight": 1}
    )
    assert edge.attrs["weight"] == 1


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

MINIMAL_CPG = {
    "treeloom_version": "0.3.0",
    "nodes": [
        {
            "id": "function:src/foo.py:10:0:1",
            "kind": "function",
            "name": "do_thing",
            "location": {"file": "src/foo.py", "line": 10, "column": 0},
            "scope": "module:src/foo.py:1:0:0",
            "attrs": {"is_async": False, "decorators": ["app.route"]},
        }
    ],
    "edges": [
        {
            "source": "function:src/foo.py:10:0:1",
            "target": "call:src/foo.py:15:4:2",
            "kind": "calls",
            "attrs": {},
        }
    ],
    "annotations": {"function:src/foo.py:10:0:1": {"summary": "does the thing"}},
    "edge_annotations": [
        {"source": "a", "target": "b", "annotations": {"label": "x"}}
    ],
}


@pytest.fixture()
def cpg_file(tmp_path):
    """Write MINIMAL_CPG to a temp file and return its Path."""
    p = tmp_path / "test.cpg.json"
    p.write_text(json.dumps(MINIMAL_CPG), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# load_cpg
# ---------------------------------------------------------------------------

def test_load_cpg_returns_cpg_data(cpg_file):
    cpg = load_cpg(cpg_file)
    assert isinstance(cpg, CpgData)


def test_load_cpg_version(cpg_file):
    cpg = load_cpg(cpg_file)
    assert cpg.treeloom_version == "0.3.0"


def test_load_cpg_nodes(cpg_file):
    cpg = load_cpg(cpg_file)
    assert len(cpg.nodes) == 1
    node = cpg.nodes[0]
    assert node.kind is NodeKind.FUNCTION
    assert node.name == "do_thing"
    assert node.location.file == "src/foo.py"
    assert node.location.line == 10
    assert node.scope == "module:src/foo.py:1:0:0"


def test_load_cpg_edges(cpg_file):
    cpg = load_cpg(cpg_file)
    assert len(cpg.edges) == 1
    edge = cpg.edges[0]
    assert edge.kind is EdgeKind.CALLS
    assert edge.source == "function:src/foo.py:10:0:1"


def test_load_cpg_annotations(cpg_file):
    cpg = load_cpg(cpg_file)
    assert cpg.annotations["function:src/foo.py:10:0:1"]["summary"] == "does the thing"
    assert len(cpg.edge_annotations) == 1


def test_load_cpg_empty_collections(tmp_path):
    """Nodes/edges/annotations keys are optional — should default to empty."""
    p = tmp_path / "empty.cpg.json"
    p.write_text(json.dumps({"treeloom_version": "0.3.0"}), encoding="utf-8")
    cpg = load_cpg(p)
    assert cpg.nodes == []
    assert cpg.edges == []
    assert cpg.annotations == {}
    assert cpg.edge_annotations == []


def test_load_cpg_node_without_location(tmp_path):
    data = {
        "treeloom_version": "0.3.0",
        "nodes": [{"id": "x", "kind": "module", "name": "main"}],
    }
    p = tmp_path / "noloc.cpg.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    cpg = load_cpg(p)
    assert cpg.nodes[0].location is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_load_cpg_bad_node_kind(tmp_path):
    data = {
        "treeloom_version": "0.3.0",
        "nodes": [{"id": "x", "kind": "spaceship", "name": "main"}],
    }
    p = tmp_path / "bad_kind.cpg.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_cpg(p)


def test_load_cpg_bad_edge_kind(tmp_path):
    data = {
        "treeloom_version": "0.3.0",
        "edges": [{"source": "a", "target": "b", "kind": "teleports_to"}],
    }
    p = tmp_path / "bad_edge.cpg.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_cpg(p)


def test_load_cpg_missing_version(tmp_path):
    p = tmp_path / "no_version.cpg.json"
    p.write_text(json.dumps({"nodes": []}), encoding="utf-8")
    with pytest.raises(KeyError):
        load_cpg(p)


def test_load_cpg_not_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("this is not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_cpg(p)


# ---------------------------------------------------------------------------
# Round-trip: construct CpgData manually, verify all fields accessible
# ---------------------------------------------------------------------------

def test_cpg_data_round_trip():
    loc = SourceLocation(file="a.py", line=1, column=0)
    node = CpgNode(id="n1", kind=NodeKind.MODULE, name="a", location=loc)
    edge = CpgEdge(source="n1", target="n2", kind=EdgeKind.CONTAINS)
    cpg = CpgData(
        treeloom_version="0.3.0",
        nodes=[node],
        edges=[edge],
        annotations={"n1": {"k": "v"}},
        edge_annotations=[{"source": "n1", "target": "n2", "annotations": {}}],
    )
    assert cpg.nodes[0].id == "n1"
    assert cpg.edges[0].kind is EdgeKind.CONTAINS
    assert cpg.annotations["n1"]["k"] == "v"
    assert cpg.edge_annotations[0]["source"] == "n1"
