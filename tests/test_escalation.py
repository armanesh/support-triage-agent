from triageagent.classify import Classification
from triageagent.escalation import decide
from triageagent.resolver import ResolutionAttempt


def _classification(**overrides):
    defaults = {"category": "billing_refund", "confidence": 0.8, "urgent": False, "negative_sentiment": False}
    defaults.update(overrides)
    return Classification(**defaults)


def _resolution(confidence=0.9):
    return ResolutionAttempt(answer="ok", confidence=confidence, tools_used=[])


def test_no_escalation_for_confident_calm_ticket():
    decision = decide(_classification(), _resolution())
    assert decision.escalate is False
    assert decision.reason is None


def test_escalates_on_negative_sentiment_regardless_of_confidence():
    decision = decide(_classification(negative_sentiment=True), _resolution(confidence=0.99))
    assert decision.escalate is True
    assert "sentiment" in decision.reason.lower()


def test_escalates_on_urgency_regardless_of_confidence():
    decision = decide(_classification(urgent=True), _resolution(confidence=0.99))
    assert decision.escalate is True
    assert "urgent" in decision.reason.lower()


def test_escalates_on_low_router_confidence():
    decision = decide(_classification(confidence=0.1), _resolution(confidence=0.99))
    assert decision.escalate is True
    assert "router confidence" in decision.reason.lower()


def test_escalates_on_low_resolver_confidence():
    decision = decide(_classification(confidence=0.8), _resolution(confidence=0.1))
    assert decision.escalate is True
    assert "resolver confidence" in decision.reason.lower()
