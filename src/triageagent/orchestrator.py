"""Wires the three agents together: router -> resolver -> escalation.

This is the multi-agent handoff itself. Each agent only knows its own job;
the orchestrator is what turns "classify, then resolve, then decide whether
to escalate" into a pipeline, and it's the one place that assembles the trace
of what happened, which is what makes a triage decision auditable after the
fact instead of a black box.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from triageagent.classify import Classifier
from triageagent.escalation import decide
from triageagent.resolver import ResolverAgent
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
        trace: list[str] = []

        classification = self.classifier.classify(ticket.subject, ticket.body)
        trace.append(
            f"router: category={classification.category} "
            f"confidence={classification.confidence:.2f} "
            f"urgent={classification.urgent} negative_sentiment={classification.negative_sentiment}"
        )

        resolution = self.resolver.resolve(ticket, classification)
        trace.append(
            f"resolver: tools={resolution.tools_used or ['none']} "
            f"confidence={resolution.confidence:.2f}"
        )

        escalation = decide(classification, resolution)
        trace.append(f"escalation: escalate={escalation.escalate} reason={escalation.reason}")

        if escalation.escalate:
            self.queue.enqueue(
                EscalationEntry(
                    ticket=ticket,
                    reason=escalation.reason or "unspecified",
                    category=classification.category,
                    confidence=resolution.confidence,
                )
            )
            response = (
                "Thanks for reaching out. This has been passed to a member of our team, "
                "who will follow up with you shortly."
            )
        else:
            response = resolution.answer

        return TriageResult(
            ticket_id=ticket.ticket_id,
            category=classification.category,
            confidence=resolution.confidence,
            escalated=escalation.escalate,
            escalation_reason=escalation.reason,
            response=response,
            trace=trace,
        )
