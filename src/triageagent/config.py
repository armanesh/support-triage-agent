"""Paths and thresholds shared across the pipeline."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KB_DIR = REPO_ROOT / "data" / "knowledge_base"
DEFAULT_TICKETS_PATH = REPO_ROOT / "data" / "sample_tickets.jsonl"
DEFAULT_QUEUE_PATH = REPO_ROOT / "data" / "escalations.jsonl"
DEFAULT_ORDERS_PATH = REPO_ROOT / "data" / "orders.json"

# Below this classifier or resolver confidence, the ticket is escalated to a
# human rather than auto-answered. Kept as named constants rather than magic
# numbers so the escalation policy is easy to audit and tune.
CATEGORY_CONFIDENCE_THRESHOLD = 0.34
RESOLVER_CONFIDENCE_THRESHOLD = 0.5
