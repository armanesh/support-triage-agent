"""The resolver agent: picks and calls tools based on the router's category,
and produces a draft answer plus a confidence score.

This is where "agentic" earns its name over a plain if/else script: the
resolver's job is to decide which combination of tools answers this specific
ticket, not to run a fixed pipeline every time. A billing ticket with no
extractable order ID gets a different (lower-confidence) path than one with a
matched order in the mock database, and that branching is what the escalation
policy downstream reacts to.
"""

from __future__ import annotations

from dataclasses import dataclass

from triageagent.classify import Classification, extract_order_id
from triageagent.ticket import Ticket
from triageagent.tools import KnowledgeBaseTool, OrderStatusTool, RefundCalculatorTool


@dataclass(frozen=True)
class ResolutionAttempt:
    answer: str
    confidence: float
    tools_used: list[str]


class ResolverAgent:
    def __init__(
        self,
        knowledge_base: KnowledgeBaseTool,
        order_status: OrderStatusTool,
        refund_calculator: RefundCalculatorTool,
    ):
        self.knowledge_base = knowledge_base
        self.order_status = order_status
        self.refund_calculator = refund_calculator

    def resolve(self, ticket: Ticket, classification: Classification) -> ResolutionAttempt:
        if classification.category == "billing_refund":
            return self._resolve_billing(ticket)
        if classification.category == "shipping":
            return self._resolve_shipping(ticket)
        return self._resolve_with_knowledge_base(ticket, classification)

    def _resolve_billing(self, ticket: Ticket) -> ResolutionAttempt:
        order_id = ticket.order_id or extract_order_id(ticket.body)
        if not order_id:
            return ResolutionAttempt(
                answer="I couldn't find an order number in this ticket to look up a refund.",
                confidence=0.2,
                tools_used=[],
            )
        order = self.order_status.lookup(order_id)
        if not order:
            return ResolutionAttempt(
                answer=f"I couldn't find order {order_id} in our system.",
                confidence=0.3,
                tools_used=["OrderStatusTool"],
            )
        decision = self.refund_calculator.calculate(order)
        if decision.eligible:
            answer = (
                f"Order {order_id}: refund of EUR {decision.refund_amount:.2f} approved. "
                f"{decision.reason}"
            )
        else:
            answer = f"Order {order_id}: refund not possible. {decision.reason}"
        return ResolutionAttempt(
            answer=answer, confidence=0.9, tools_used=["OrderStatusTool", "RefundCalculatorTool"]
        )

    def _resolve_shipping(self, ticket: Ticket) -> ResolutionAttempt:
        order_id = ticket.order_id or extract_order_id(ticket.body)
        if not order_id:
            return ResolutionAttempt(
                answer="I couldn't find an order number in this ticket to check shipping status.",
                confidence=0.2,
                tools_used=[],
            )
        order = self.order_status.lookup(order_id)
        if not order:
            return ResolutionAttempt(
                answer=f"I couldn't find order {order_id} in our system.",
                confidence=0.3,
                tools_used=["OrderStatusTool"],
            )
        answer = f"Order {order_id} status: {order.status}."
        return ResolutionAttempt(answer=answer, confidence=0.85, tools_used=["OrderStatusTool"])

    def _resolve_with_knowledge_base(
        self, ticket: Ticket, classification: Classification
    ) -> ResolutionAttempt:
        results = self.knowledge_base.search(f"{ticket.subject} {ticket.body}", k=2)
        if not results:
            return ResolutionAttempt(
                answer="I don't have information about this in our policy documents.",
                confidence=0.15,
                tools_used=["KnowledgeBaseTool"],
            )
        top = results[0]
        answer = f"Based on our {top.doc_id.replace('-', ' ')} policy: {top.text}"
        # confidence blends how well the KB matched with how confident the
        # router was about the category in the first place
        confidence = min(0.95, 0.4 + 0.5 * top.score + 0.1 * classification.confidence)
        return ResolutionAttempt(answer=answer, confidence=confidence, tools_used=["KnowledgeBaseTool"])
