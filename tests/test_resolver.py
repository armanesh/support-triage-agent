import pytest

from triageagent.classify import Classification
from triageagent.config import DEFAULT_KB_DIR, DEFAULT_ORDERS_PATH
from triageagent.resolver import ResolverAgent
from triageagent.ticket import Ticket
from triageagent.tools import KnowledgeBaseTool, OrderStatusTool, RefundCalculatorTool


@pytest.fixture
def resolver():
    return ResolverAgent(
        knowledge_base=KnowledgeBaseTool(DEFAULT_KB_DIR),
        order_status=OrderStatusTool(DEFAULT_ORDERS_PATH),
        refund_calculator=RefundCalculatorTool(),
    )


def _classification(category="billing_refund", confidence=0.8):
    return Classification(category=category, confidence=confidence, urgent=False, negative_sentiment=False)


def test_resolve_billing_with_valid_order(resolver):
    ticket = Ticket(ticket_id="t1", customer_name="A", subject="refund", body="please refund ORD-1001", order_id="ORD-1001")
    attempt = resolver.resolve(ticket, _classification())
    assert "refund" in attempt.answer.lower()
    assert "OrderStatusTool" in attempt.tools_used
    assert "RefundCalculatorTool" in attempt.tools_used
    assert attempt.confidence > 0.5


def test_resolve_billing_without_order_id_is_low_confidence(resolver):
    ticket = Ticket(ticket_id="t2", customer_name="A", subject="refund", body="please refund me", order_id=None)
    attempt = resolver.resolve(ticket, _classification())
    assert attempt.confidence < 0.5
    assert attempt.tools_used == []


def test_resolve_billing_with_unknown_order(resolver):
    ticket = Ticket(ticket_id="t3", customer_name="A", subject="refund", body="refund ORD-9999", order_id="ORD-9999")
    attempt = resolver.resolve(ticket, _classification())
    assert "couldn't find order" in attempt.answer.lower()


def test_resolve_shipping_with_valid_order(resolver):
    ticket = Ticket(ticket_id="t4", customer_name="A", subject="status", body="track ORD-1002", order_id="ORD-1002")
    attempt = resolver.resolve(ticket, _classification(category="shipping"))
    assert "in transit" in attempt.answer


def test_resolve_general_uses_knowledge_base(resolver):
    ticket = Ticket(ticket_id="t5", customer_name="A", subject="crash", body="the app crashes on launch")
    attempt = resolver.resolve(ticket, _classification(category="technical", confidence=0.5))
    assert "technical faq" in attempt.answer.lower()
    assert "KnowledgeBaseTool" in attempt.tools_used


def test_resolve_general_with_no_kb_match_is_low_confidence(resolver):
    ticket = Ticket(ticket_id="t6", customer_name="A", subject="xyz", body="qwertyuiop asdfgh")
    attempt = resolver.resolve(ticket, _classification(category="general", confidence=0.0))
    assert attempt.confidence < 0.3
