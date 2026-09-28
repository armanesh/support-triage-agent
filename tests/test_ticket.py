from triageagent.config import DEFAULT_TICKETS_PATH
from triageagent.ticket import EscalationEntry, Ticket, TicketQueue, load_tickets


def test_load_tickets_reads_the_real_sample_file():
    tickets = load_tickets(DEFAULT_TICKETS_PATH)
    assert len(tickets) >= 5
    assert all(t.ticket_id and t.body for t in tickets)


def test_queue_starts_empty(tmp_path):
    queue = TicketQueue(tmp_path / "q.jsonl")
    assert queue.list() == []


def test_queue_enqueue_and_list_round_trip(tmp_path):
    queue = TicketQueue(tmp_path / "q.jsonl")
    ticket = Ticket(ticket_id="t1", customer_name="A", subject="s", body="b", order_id="ORD-1")
    entry = EscalationEntry(ticket=ticket, reason="test reason", category="billing_refund", confidence=0.2)
    queue.enqueue(entry)

    loaded = queue.list()
    assert len(loaded) == 1
    assert loaded[0].ticket == ticket
    assert loaded[0].reason == "test reason"


def test_queue_clear_removes_file(tmp_path):
    path = tmp_path / "q.jsonl"
    queue = TicketQueue(path)
    ticket = Ticket(ticket_id="t1", customer_name="A", subject="s", body="b")
    queue.enqueue(EscalationEntry(ticket=ticket, reason="r", category="general", confidence=0.1))
    assert path.exists()
    queue.clear()
    assert not path.exists()
    assert queue.list() == []
