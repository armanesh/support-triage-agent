"""Command-line entry point: `triage <command> [options]`."""

from __future__ import annotations

import argparse
import uuid

from triageagent.classify import get_classifier
from triageagent.config import (
    DEFAULT_KB_DIR,
    DEFAULT_ORDERS_PATH,
    DEFAULT_QUEUE_PATH,
    DEFAULT_TICKETS_PATH,
)
from triageagent.orchestrator import TriageOrchestrator
from triageagent.resolver import ResolverAgent
from triageagent.ticket import Ticket, TicketQueue, load_tickets
from triageagent.tools import KnowledgeBaseTool, OrderStatusTool, RefundCalculatorTool


def _build_orchestrator(args: argparse.Namespace) -> TriageOrchestrator:
    classifier = get_classifier(args.classifier)
    resolver = ResolverAgent(
        knowledge_base=KnowledgeBaseTool(args.kb_dir),
        order_status=OrderStatusTool(args.orders),
        refund_calculator=RefundCalculatorTool(),
    )
    queue = TicketQueue(args.queue)
    return TriageOrchestrator(classifier, resolver, queue)


def _print_result(result) -> None:
    print(f"[{result.ticket_id}] category={result.category} escalated={result.escalated}")
    for line in result.trace:
        print(f"    {line}")
    print(f"    -> {result.response}\n")


def cmd_run(args: argparse.Namespace) -> int:
    orchestrator = _build_orchestrator(args)
    tickets = load_tickets(args.tickets)
    escalated_count = 0
    for ticket in tickets:
        result = orchestrator.handle(ticket)
        _print_result(result)
        if result.escalated:
            escalated_count += 1
    print(f"Processed {len(tickets)} tickets, {escalated_count} escalated to a human.")
    return 0


def cmd_handle(args: argparse.Namespace) -> int:
    orchestrator = _build_orchestrator(args)
    ticket = Ticket(
        ticket_id=args.ticket_id or f"adhoc-{uuid.uuid4().hex[:8]}",
        customer_name=args.customer or "Anonymous",
        subject=args.subject or "",
        body=args.body,
        order_id=args.order_id,
    )
    result = orchestrator.handle(ticket)
    _print_result(result)
    return 0


def cmd_show_queue(args: argparse.Namespace) -> int:
    queue = TicketQueue(args.queue)
    entries = queue.list()
    if not entries:
        print("Escalation queue is empty.")
        return 0
    for entry in entries:
        print(f"[{entry.ticket.ticket_id}] {entry.category} (confidence {entry.confidence:.2f})")
        print(f"    reason: {entry.reason}")
        print(f"    from: {entry.ticket.customer_name} — {entry.ticket.subject}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="triage", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument("--kb-dir", default=str(DEFAULT_KB_DIR))
    common_args.add_argument("--orders", default=str(DEFAULT_ORDERS_PATH))
    common_args.add_argument("--queue", default=str(DEFAULT_QUEUE_PATH))
    common_args.add_argument("--classifier", choices=["rule-based", "openai"], default="rule-based")

    p = sub.add_parser("run", parents=[common_args], help="Process a batch of tickets from a JSONL file")
    p.add_argument("--tickets", default=str(DEFAULT_TICKETS_PATH))
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("handle", parents=[common_args], help="Process a single ad-hoc ticket")
    p.add_argument("body", help="The ticket body text")
    p.add_argument("--subject", default="")
    p.add_argument("--customer", default=None)
    p.add_argument("--order-id", default=None)
    p.add_argument("--ticket-id", default=None)
    p.set_defaults(func=cmd_handle)

    p = sub.add_parser("show-queue", help="Print the current human escalation queue")
    p.add_argument("--queue", default=str(DEFAULT_QUEUE_PATH))
    p.set_defaults(func=cmd_show_queue)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
