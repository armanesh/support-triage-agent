"""The escalation agent: the last checkpoint before a ticket is auto-answered.

Kept as its own module, separate from the resolver, because escalation is a
policy decision (what confidence is acceptable to auto-answer at, which
signals override confidence entirely) rather than a resolution step. Keeping
it separate also means the thresholds in config.py can be tuned, or this
whole function replaced with an ML-based risk model later, without touching
how tickets get classified or resolved.
"""

from __future__ import annotations

from dataclasses import dataclass

from triageagent.classify import Classification
from triageagent.config import CATEGORY_CONFIDENCE_THRESHOLD, RESOLVER_CONFIDENCE_THRESHOLD
from triageagent.resolver import ResolutionAttempt


@dataclass(frozen=True)
class EscalationDecision:
    escalate: bool
    reason: str | None


def decide(classification: Classification, resolution: ResolutionAttempt) -> EscalationDecision:
    if classification.negative_sentiment:
        return EscalationDecision(True, "Negative sentiment detected; routing to a human.")
    if classification.urgent:
        return EscalationDecision(True, "Ticket flagged urgent; routing to a human.")
    if classification.confidence < CATEGORY_CONFIDENCE_THRESHOLD:
        return EscalationDecision(
            True,
            f"Router confidence {classification.confidence:.2f} below threshold "
            f"{CATEGORY_CONFIDENCE_THRESHOLD}.",
        )
    if resolution.confidence < RESOLVER_CONFIDENCE_THRESHOLD:
        return EscalationDecision(
            True,
            f"Resolver confidence {resolution.confidence:.2f} below threshold "
            f"{RESOLVER_CONFIDENCE_THRESHOLD}.",
        )
    return EscalationDecision(False, None)
