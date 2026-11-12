from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from triageagent.classify import Classification, Classifier, RuleBasedClassifier
from triageagent.config import DEFAULT_KB_DIR, DEFAULT_ORDERS_PATH
from triageagent.orchestrator import TriageOrchestrator
from triageagent.resolver import ResolverAgent
from triageagent.ticket import Ticket, TicketQueue
from triageagent.tools import KnowledgeBaseTool, OrderStatusTool, RefundCalculatorTool


def _resolver() -> ResolverAgent:
    return ResolverAgent(
        knowledge_base=KnowledgeBaseTool(DEFAULT_KB_DIR),
        order_status=OrderStatusTool(DEFAULT_ORDERS_PATH),
        refund_calculator=RefundCalculatorTool(),
    )


def _tickets(n: int) -> list[Ticket]:
    return [
        Ticket(
            ticket_id=f"t{i}",
            customer_name="A",
            subject="refund",
            body="refund please ORD-1001",
            order_id="ORD-1001",
        )
        for i in range(n)
    ]


class SlowClassifier(Classifier):
    """Async classifier that sleeps, like a network call, and records how many
    calls were in flight at the same time."""

    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self.in_flight = 0
        self.max_in_flight = 0
        self._inner = RuleBasedClassifier()

    def classify(self, subject: str, body: str) -> Classification:
        return self._inner.classify(subject, body)

    async def aclassify(self, subject: str, body: str) -> Classification:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(self.delay)
        self.in_flight -= 1
        return self._inner.classify(subject, body)


def test_ahandle_matches_sync_handle(tmp_path: Path) -> None:
    ticket = _tickets(1)[0]
    sync_orch = TriageOrchestrator(RuleBasedClassifier(), _resolver(), TicketQueue(tmp_path / "a.jsonl"))
    async_orch = TriageOrchestrator(RuleBasedClassifier(), _resolver(), TicketQueue(tmp_path / "b.jsonl"))
    assert asyncio.run(async_orch.ahandle(ticket)) == sync_orch.handle(ticket)


def test_default_aclassify_wraps_sync_classify() -> None:
    classifier = RuleBasedClassifier()
    expected = classifier.classify("refund", "refund ORD-1001")
    assert asyncio.run(classifier.aclassify("refund", "refund ORD-1001")) == expected


def test_batch_runs_tickets_concurrently(tmp_path: Path) -> None:
    classifier = SlowClassifier(delay=0.05)
    orch = TriageOrchestrator(classifier, _resolver(), TicketQueue(tmp_path / "q.jsonl"))
    start = time.perf_counter()
    results = asyncio.run(orch.handle_batch(_tickets(10), concurrency=10))
    elapsed = time.perf_counter() - start
    assert len(results) == 10
    assert classifier.max_in_flight == 10
    assert elapsed < 0.3  # sequential would take at least 0.5s


def test_batch_respects_concurrency_limit(tmp_path: Path) -> None:
    classifier = SlowClassifier(delay=0.02)
    orch = TriageOrchestrator(classifier, _resolver(), TicketQueue(tmp_path / "q.jsonl"))
    asyncio.run(orch.handle_batch(_tickets(12), concurrency=3))
    assert classifier.max_in_flight == 3


def test_batch_preserves_input_order(tmp_path: Path) -> None:
    orch = TriageOrchestrator(SlowClassifier(), _resolver(), TicketQueue(tmp_path / "q.jsonl"))
    tickets = _tickets(8)
    results = asyncio.run(orch.handle_batch(tickets, concurrency=4))
    assert [r.ticket_id for r in results] == [t.ticket_id for t in tickets]


def test_concurrent_escalations_write_whole_rows(tmp_path: Path) -> None:
    queue = TicketQueue(tmp_path / "q.jsonl")
    orch = TriageOrchestrator(RuleBasedClassifier(), _resolver(), queue)
    angry = [
        Ticket(ticket_id=f"a{i}", customer_name="A", subject="angry", body="this is a scam, unacceptable")
        for i in range(20)
    ]
    results = asyncio.run(orch.handle_batch(angry, concurrency=10))
    assert all(r.escalated for r in results)
    assert sorted(e.ticket.ticket_id for e in queue.list()) == sorted(t.ticket_id for t in angry)


def test_batch_rejects_bad_concurrency(tmp_path: Path) -> None:
    orch = TriageOrchestrator(RuleBasedClassifier(), _resolver(), TicketQueue(tmp_path / "q.jsonl"))
    with pytest.raises(ValueError):
        asyncio.run(orch.handle_batch(_tickets(1), concurrency=0))
