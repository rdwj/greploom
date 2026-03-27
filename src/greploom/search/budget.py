"""Token budget management: format ExpandedNodes into LLM-friendly context blocks."""

from __future__ import annotations

from dataclasses import dataclass, field

import tiktoken

from greploom.cpg_types import NodeKind
from greploom.search.expand import ExpandedNode

# Lazy singleton — initialized on first use.
_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def _count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    enc = _get_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def _format_summary(node_id: str, kind: NodeKind, name: str, attrs: dict) -> str:
    """Generate a concise summary line from CPG node data."""
    if kind is NodeKind.FUNCTION:
        params = attrs.get("params", attrs.get("parameters", ""))
        param_str = params if isinstance(params, str) else ", ".join(params) if params else ""
        return f"function {name}({param_str})"
    if kind is NodeKind.CLASS:
        return f"class {name}"
    if kind is NodeKind.MODULE:
        return f"module {name}"
    return f"{kind.value} {name}"


def _format_block(expanded: ExpandedNode) -> str:
    node = expanded.node
    loc = node.location
    file_str = loc.file if loc else None
    line_str = loc.line if loc else None

    loc_part = f"({file_str}:{line_str})" if file_str and line_str else (file_str or "")
    header = f"## {expanded.relationship}: {node.kind.value} {node.name} {loc_part}".rstrip()

    summary = _format_summary(node.id, node.kind, node.name, node.attrs)
    body_lines = [summary]
    if loc and loc.file and loc.line:
        body_lines.append(f"Defined at {loc.file}:{loc.line}")

    return header + "\n\n" + "\n".join(body_lines)


@dataclass
class ContextBlock:
    node_id: str
    file: str | None
    line: int | None
    name: str
    kind: str
    relationship: str
    text: str
    tokens: int


@dataclass
class ContextResult:
    blocks: list[ContextBlock] = field(default_factory=list)
    total_tokens: int = 0
    budget: int = 8192
    truncated: bool = False


def assemble_context(
    expanded: list[ExpandedNode],
    budget: int = 8192,
) -> ContextResult:
    """Format and pack ExpandedNodes into a token-budgeted ContextResult."""
    result = ContextResult(budget=budget)
    if not expanded:
        return result

    # expand_hits already sorts by relevance, but guarantee it here.
    # Match expand_hits sort: relevance descending, then relationship priority.
    _rel_pri = {"hit": 0, "caller": 1, "callee": 2, "class": 3, "parameter": 4,
                "data_source": 5, "import": 6}
    ordered = sorted(expanded, key=lambda e: (-e.relevance, _rel_pri.get(e.relationship, 99)))
    remaining = budget

    for exp in ordered:
        text = _format_block(exp)
        count = _count_tokens(text)
        node = exp.node
        loc = node.location

        if count > remaining:
            if remaining <= 0:
                result.truncated = True
                continue
            text = _truncate_to_tokens(text, remaining)
            count = _count_tokens(text)
            result.truncated = True

        result.blocks.append(ContextBlock(
            node_id=node.id,
            file=loc.file if loc else None,
            line=loc.line if loc else None,
            name=node.name,
            kind=node.kind.value,
            relationship=exp.relationship,
            text=text,
            tokens=count,
        ))
        result.total_tokens += count
        remaining -= count

    return result
