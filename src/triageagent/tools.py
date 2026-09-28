"""Tools the resolver agent can call.

Each tool is a small, independently testable unit with a narrow job, which is
the point of tool-use in an agent system: the agent's job is deciding *which*
tool to call and *how to combine* the results, not doing the lookup or
calculation itself. Three tools, three different flavors of "tool":

- KnowledgeBaseTool: retrieval over static policy documents.
- OrderStatusTool: a lookup against structured state (a mocked orders DB here;
  a real order-management API in production).
- RefundCalculatorTool: a pure business-rule function with no external state
  at all.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

TOKEN_RE = re.compile(r"[a-zA-Z]+")


def _tokenize(text: str) -> set[str]:
    return {tok.lower() for tok in TOKEN_RE.findall(text)}


@dataclass(frozen=True)
class KBResult:
    doc_id: str
    heading: str
    text: str
    score: float


class KnowledgeBaseTool:
    """Keyword-overlap search over the markdown policy documents.

    Deliberately simpler than the embedding-based retrieval in the companion
    nl-tax-rag-assistant project: this repo's point is agent orchestration and
    tool use, not retrieval quality, so a dependency-free keyword overlap
    scorer keeps the tool auditable in a couple of lines while still being a
    real, working retrieval step.
    """

    def __init__(self, kb_dir: str | Path):
        self.documents: list[tuple[str, str, str]] = []  # (doc_id, heading, body)
        for path in sorted(Path(kb_dir).glob("*.md")):
            doc_id = path.stem
            for heading, body in _split_sections(path.read_text(encoding="utf-8")):
                self.documents.append((doc_id, heading, body))

    def search(self, query: str, k: int = 2) -> list[KBResult]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        scored = []
        for doc_id, heading, body in self.documents:
            body_tokens = _tokenize(f"{heading} {body}")
            overlap = query_tokens & body_tokens
            score = len(overlap) / len(query_tokens)
            if score > 0:
                scored.append(KBResult(doc_id=doc_id, heading=heading, text=body, score=score))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]


def _split_sections(text: str) -> list[tuple[str, str]]:
    heading_re = re.compile(r"^#{1,3}\s+(.*)$", re.MULTILINE)
    matches = list(heading_re.finditer(text))
    if not matches:
        return [("", text.strip())]
    sections = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((match.group(1).strip(), body))
    return sections


@dataclass(frozen=True)
class OrderInfo:
    order_id: str
    status: str
    days_since_purchase: int
    amount_eur: float


class OrderStatusTool:
    def __init__(self, orders_path: str | Path):
        raw = json.loads(Path(orders_path).read_text(encoding="utf-8"))
        self._orders = {
            order_id: OrderInfo(order_id=order_id, **fields) for order_id, fields in raw.items()
        }

    def lookup(self, order_id: str) -> OrderInfo | None:
        return self._orders.get(order_id)


@dataclass(frozen=True)
class RefundDecision:
    eligible: bool
    refund_amount: float
    reason: str


class RefundCalculatorTool:
    """Pure business-rule function: no I/O, trivial to unit test exhaustively.

    Full refund within 30 days of purchase, half refund between 30 and 60
    days, no refund after that. A real system would source this rule from a
    policy configuration rather than hardcoding it; it's inlined here so the
    logic is visible in one place for a portfolio review.
    """

    def calculate(self, order: OrderInfo) -> RefundDecision:
        if order.days_since_purchase <= 30:
            return RefundDecision(
                eligible=True,
                refund_amount=order.amount_eur,
                reason="Within 30-day full refund window.",
            )
        if order.days_since_purchase <= 60:
            return RefundDecision(
                eligible=True,
                refund_amount=round(order.amount_eur * 0.5, 2),
                reason="Within 31-60 day window: 50% refund per policy.",
            )
        return RefundDecision(
            eligible=False,
            refund_amount=0.0,
            reason="Purchase was more than 60 days ago; outside refund window.",
        )
