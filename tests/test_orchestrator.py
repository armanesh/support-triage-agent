from triageagent.classify import RuleBasedClassifier
from triageagent.config import DEFAULT_KB_DIR, DEFAULT_ORDERS_PATH
from triageagent.orchestrator import TriageOrchestrator
from triageagent.resolver import ResolverAgent
from triageagent.ticket import Ticket, TicketQueue
from triageagent.tools import KnowledgeBaseTool, OrderStatusTool, RefundCalculatorTool


def _orchestrator(tmp_path):
    resolver = ResolverAgent(
        knowledge_base=KnowledgeBaseTool(DEFAULT_KB_DIR),
        order_status=OrderStatusTool(DEFAULT_ORDERS_PATH),
        refund_calculator=RefundCalculatorTool(),
    )
    queue = TicketQueue(tmp_path / "escalations.jsonl")
    return TriageOrchestrator(RuleBasedClassifier(), resolver, queue), queue


def test_clean_refund_request_is_auto_resolved_not_escalated(tmp_path):
    orchestrator, queue = _orchestrator(tmp_path)
    ticket = Ticket(ticket_id="t1", customer_name="A", subject="refund", body="refund please ORD-1001", order_id="ORD-1001")
    result = orchestrator.handle(ticket)
    assert result.escalated is False
    assert "refund" in result.response.lower()
    assert queue.list() == []


def test_angry_ticket_is_escalated_and_queued(tmp_path):
    orchestrator, queue = _orchestrator(tmp_path)
    ticket = Ticket(
        ticket_id="t2",
        customer_name="A",
        subject="furious",
        body="I am furious this is a scam and unacceptable",
        order_id=None,
    )
    result = orchestrator.handle(ticket)
    assert result.escalated is True
    queued = queue.list()
    assert len(queued) == 1
    assert queued[0].ticket.ticket_id == "t2"


def test_urgent_ticket_is_escalated(tmp_path):
    orchestrator, _ = _orchestrator(tmp_path)
    ticket = Ticket(ticket_id="t3", customer_name="A", subject="urgent", body="need this fixed immediately")
    result = orchestrator.handle(ticket)
    assert result.escalated is True


def test_trace_records_each_agent_step(tmp_path):
    orchestrator, _ = _orchestrator(tmp_path)
    ticket = Ticket(ticket_id="t4", customer_name="A", subject="refund", body="refund ORD-1001", order_id="ORD-1001")
    result = orchestrator.handle(ticket)
    assert any(line.startswith("router:") for line in result.trace)
    assert any(line.startswith("resolver:") for line in result.trace)
    assert any(line.startswith("escalation:") for line in result.trace)


def test_vague_ticket_escalates_on_low_confidence(tmp_path):
    orchestrator, _ = _orchestrator(tmp_path)
    ticket = Ticket(ticket_id="t5", customer_name="A", subject="hi", body="things are weird with my stuff")
    result = orchestrator.handle(ticket)
    assert result.escalated is True
