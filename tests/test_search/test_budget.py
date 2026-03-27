"""Tests for greploom.search.budget — token budget assembly."""

from __future__ import annotations

from greploom.cpg_types import CpgNode, NodeKind, SourceLocation
from greploom.search.budget import _count_tokens, assemble_context
from greploom.search.expand import ExpandedNode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(
    name: str,
    kind: NodeKind = NodeKind.FUNCTION,
    *,
    nid: str | None = None,
    file: str | None = "src/foo.py",
    line: int | None = 10,
    attrs: dict | None = None,
) -> CpgNode:
    loc = SourceLocation(file=file, line=line) if file and line else None
    return CpgNode(
        id=nid or name,
        kind=kind,
        name=name,
        location=loc,
        attrs=attrs or {},
    )


def _exp(name: str, relevance: float = 1.0, relationship: str = "hit", **kwargs) -> ExpandedNode:
    return ExpandedNode(node=_node(name, **kwargs), relevance=relevance, relationship=relationship)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty_result():
    result = assemble_context([])
    assert result.blocks == []
    assert result.total_tokens == 0
    assert result.truncated is False


def test_all_blocks_fit_within_budget():
    nodes = [_exp("alpha", 1.0), _exp("beta", 0.8), _exp("gamma", 0.6)]
    result = assemble_context(nodes, budget=500)

    assert result.truncated is False
    assert len(result.blocks) == 3
    assert result.total_tokens <= 500
    # total_tokens must match actual sum of block tokens.
    assert result.total_tokens == sum(b.tokens for b in result.blocks)


def test_total_tokens_accurate():
    nodes = [_exp("compute", 1.0)]
    result = assemble_context(nodes, budget=500)

    assert len(result.blocks) == 1
    block = result.blocks[0]
    assert block.tokens == _count_tokens(block.text)
    assert result.total_tokens == block.tokens


def test_high_relevance_blocks_included_when_budget_tight():
    """With a tight budget, highest-relevance nodes must appear in output."""
    nodes = [
        _exp("low_rel", 0.1, relationship="import"),
        _exp("high_rel", 1.0, relationship="hit"),
        _exp("mid_rel", 0.5, relationship="callee"),
    ]
    # Budget sized to fit roughly one block.
    result = assemble_context(nodes, budget=50)

    included_names = {b.name for b in result.blocks}
    # The highest-relevance node should always be included.
    assert "high_rel" in included_names
    assert result.truncated is True


def test_single_block_truncated_to_fit():
    nodes = [_exp("big_func", 1.0)]
    # Generate one block and learn its natural size.
    full_result = assemble_context(nodes, budget=10_000)
    assert len(full_result.blocks) == 1
    natural_tokens = full_result.blocks[0].tokens

    # Budget smaller than the block.
    small_budget = max(1, natural_tokens // 2)
    result = assemble_context(nodes, budget=small_budget)

    assert result.truncated is True
    assert len(result.blocks) == 1
    assert result.blocks[0].tokens <= small_budget
    assert result.total_tokens <= small_budget


def test_budget_zero_produces_empty_blocks():
    nodes = [_exp("something", 1.0)]
    result = assemble_context(nodes, budget=0)

    assert result.blocks == []
    assert result.total_tokens == 0
    assert result.truncated is True


def test_some_blocks_dropped_sets_truncated():
    """Ensure truncated=True when nodes are skipped entirely (not just shrunk)."""
    # Build a list where the first two fit but the rest cannot.
    nodes = [_exp(f"func_{i}", relevance=1.0 - i * 0.01) for i in range(10)]
    full_result = assemble_context(nodes, budget=10_000)
    natural_total = full_result.total_tokens

    # Budget that fits roughly half the blocks.
    half_budget = natural_total // 2
    result = assemble_context(nodes, budget=half_budget)

    assert result.truncated is True
    assert result.total_tokens <= half_budget
    # First (highest-relevance) block must be present.
    assert result.blocks[0].name == "func_0"


def test_block_fields_populated_correctly():
    node = _exp("my_func", 1.0, relationship="callee", file="lib/bar.py", line=42)
    result = assemble_context([node], budget=500)

    assert len(result.blocks) == 1
    b = result.blocks[0]
    assert b.name == "my_func"
    assert b.kind == "function"
    assert b.relationship == "callee"
    assert b.file == "lib/bar.py"
    assert b.line == 42
    assert "callee" in b.text
    assert "my_func" in b.text


def test_node_without_location():
    node = ExpandedNode(
        node=CpgNode(id="no_loc", kind=NodeKind.VARIABLE, name="x"),
        relevance=1.0,
        relationship="hit",
    )
    result = assemble_context([node], budget=200)
    assert len(result.blocks) == 1
    b = result.blocks[0]
    assert b.file is None
    assert b.line is None


def test_class_and_module_formatting():
    nodes = [
        _exp("MyClass", 0.9, relationship="class", kind=NodeKind.CLASS),
        _exp("mymodule", 0.7, relationship="import", kind=NodeKind.MODULE),
    ]
    result = assemble_context(nodes, budget=500)
    texts = {b.name: b.text for b in result.blocks}
    assert "class MyClass" in texts["MyClass"]
    assert "module mymodule" in texts["mymodule"]
