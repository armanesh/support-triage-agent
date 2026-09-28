"""Ticket data model and the file-backed escalation queue.

The escalation queue is deliberately just an append-only JSONL file, not a
database. This is a portfolio project demonstrating agent orchestration and
escalation logic, not a queueing system; a real deployment would swap this
for a proper ticketing system's API (Zendesk, Freshdesk, an internal queue),
and nothing else in the pipeline would need to change since TicketQueue's
interface (enqueue/list) is what the rest of the code depends on.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    customer_name: str
    subject: str
    body: str
    order_id: str | None = None


@dataclass
class EscalationEntry:
    ticket: Ticket
    reason: str
    category: str
    confidence: float


class TicketQueue:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def enqueue(self, entry: EscalationEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ticket": asdict(entry.ticket),
            "reason": entry.reason,
            "category": entry.category,
            "confidence": entry.confidence,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def list(self) -> list[EscalationEntry]:
        if not self.path.exists():
            return []
        entries = []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                entries.append(
                    EscalationEntry(
                        ticket=Ticket(**row["ticket"]),
                        reason=row["reason"],
                        category=row["category"],
                        confidence=row["confidence"],
                    )
                )
        return entries

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()


def load_tickets(path: str | Path) -> list[Ticket]:
    tickets = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tickets.append(Ticket(**row))
    return tickets
