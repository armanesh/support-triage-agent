import json

import pytest

from triageagent.config import DEFAULT_KB_DIR, DEFAULT_ORDERS_PATH
from triageagent.tools import KnowledgeBaseTool, OrderStatusTool, RefundCalculatorTool


def test_knowledge_base_finds_refund_policy():
    kb = KnowledgeBaseTool(DEFAULT_KB_DIR)
    results = kb.search("how do refunds work", k=2)
    assert results
    assert results[0].doc_id == "refund-policy"


def test_knowledge_base_returns_empty_for_empty_query():
    kb = KnowledgeBaseTool(DEFAULT_KB_DIR)
    assert kb.search("", k=2) == []


def test_order_status_tool_finds_known_order():
    tool = OrderStatusTool(DEFAULT_ORDERS_PATH)
    order = tool.lookup("ORD-1001")
    assert order is not None
    assert order.status == "delivered"


def test_order_status_tool_returns_none_for_unknown_order():
    tool = OrderStatusTool(DEFAULT_ORDERS_PATH)
    assert tool.lookup("ORD-9999") is None


def test_order_status_tool_loads_custom_file(tmp_path):
    orders_path = tmp_path / "orders.json"
    orders_path.write_text(
        json.dumps({"ORD-X": {"status": "delivered", "days_since_purchase": 1, "amount_eur": 10.0}}),
        encoding="utf-8",
    )
    tool = OrderStatusTool(orders_path)
    order = tool.lookup("ORD-X")
    assert order.amount_eur == 10.0


@pytest.mark.parametrize(
    "days,expected_eligible,expected_fraction",
    [
        (5, True, 1.0),
        (30, True, 1.0),
        (31, True, 0.5),
        (60, True, 0.5),
        (61, False, 0.0),
        (200, False, 0.0),
    ],
)
def test_refund_calculator_windows(days, expected_eligible, expected_fraction):
    from triageagent.tools import OrderInfo

    order = OrderInfo(order_id="ORD-T", status="delivered", days_since_purchase=days, amount_eur=100.0)
    decision = RefundCalculatorTool().calculate(order)
    assert decision.eligible == expected_eligible
    assert decision.refund_amount == pytest.approx(100.0 * expected_fraction)
