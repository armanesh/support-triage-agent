"""Wires the three agents together: router -> resolver -> escalation.

This is the multi-agent handoff itself. Each agent only knows its own job;
the orchestrator is what turns "classify, then resolve, then decide whether
to escalate" into a pipeline, and it's the one place that assembles the trace
of what happened, which is what makes a triage decision auditable after the
fact instead of a black box.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field

from triageagent.classify import Classification, Classifier
from triageagent.escalation import decide
from triageagent.resolver import ResolutionAttempt, ResolverAgent
from triageagent.ticket import EscalationEntry, Ticket, TicketQueue


@dataclass
class TriageResult:
    ticket_id: str
    category: str
    confidence: float
    escalated: bool
    escalation_reason: str | None
    response: str
    trace: list[str] = field(default_factory=list)


class TriageOrchestrator:
    def __init__(self, classifier: Classifier, resolver: ResolverAgent, queue: TicketQueue):
        self.classifier = classifier
        self.resolver = resolver
        self.queue = queue

    def handle(self, ticket: Ticket) -> TriageResult:
        classification = self.classifier.classify(ticket.subject, ticket.body)
        resolution = self.resolver.resolve(ticket, classification)
        result, entry = self._decide(ticket, classification, resolution)
        if entry is not None:
            self.queue.enqueue(entry)
        return result

    async def ahandle(self, ticket: Ticket) -> TriageResult:
        """Same pipeline as `handle`, but every agent step is awaited, so many
        tickets can be in flight at once."""
        classification = await self.classifier.aclassify(ticket.subject, ticket.body)
        resolution = await self.resolver.aresolve(ticket, classification)
        result, entry = self._decide(ticket, classification, resolution)
        if entry is not None:
            await self.queue.aenqueue(entry)
        return result

    async def handle_batch(
        self, tickets: Sequence[Ticket], concurrency: int = 5
    ) -> list[TriageResult]:
        """Process tickets concurrently, at most `concurrency` at a time.

        The semaphore caps how many tickets (and so how many LLM calls) run at
        once. Results come back in the same order as `tickets`.
        """
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        semaphore = asyncio.Semaphore(concurrency)

        async def worker(ticket: Ticket) -> TriageResult:
            async with semaphore:
                return await self.ahandle(ticket)

        return list(await asyncio.gather(*(worker(t) for t in tickets)))

    @staticmethod
    def _decide(
        ticket: Ticket, classification: Classification, resolution: ResolutionAttempt
    ) -> tuple[TriageResult, EscalationEntry | None]:
        """Pure step shared by the sync and async paths: build the trace, apply
        the escalation policy, and say whether the ticket needs queueing."""
        trace: list[str] = [
            (
                f"router: category={classification.category} "
                f"confidence={classification.confidence:.2f} "
                f"urgent={classification.urgent} "
                f"negative_sentiment={classification.negative_sentiment}"
            ),
            (
                f"resolver: tools={resolution.tools_used or ['none']} "
                f"confidence={resolution.confidence:.2f}"
            ),
        ]

        escalation = decide(classification, resolution)
        trace.append(f"escalation: escalate={escalation.escalate} reason={escalation.reason}")

        entry: EscalationEntry | None = None
        if escalation.escalate:
            entry = EscalationEntry(
                ticket=ticket,
                reason=escalation.reason or "unspecified",
                category=classification.category,
                confidence=resolution.confidence,
            )
            response = (
                "Thanks for reaching out. This has been passed to a member of our team, "
                "who will follow up with you shortly."
            )
        else:
            response = resolution.answer

        result = TriageResult(
            ticket_id=ticket.ticket_id,
            category=classification.category,
            confidence=resolution.confidence,
            escalated=escalation.escalate,
            escalation_reason=escalation.reason,
            response=response,
            trace=trace,
        )
        return result, entry
