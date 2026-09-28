from triageagent.classify import RuleBasedClassifier, extract_order_id


def test_classifies_refund_request_as_billing():
    result = RuleBasedClassifier().classify("Refund", "I'd like a refund for my overcharged invoice")
    assert result.category == "billing_refund"
    assert result.confidence > 0


def test_classifies_shipping_question():
    result = RuleBasedClassifier().classify("Where's my order", "tracking shows no delivery update")
    assert result.category == "shipping"


def test_classifies_account_lockout():
    result = RuleBasedClassifier().classify("locked out", "I forgot my password and can't login")
    assert result.category == "account"


def test_detects_urgency_keyword():
    result = RuleBasedClassifier().classify("help", "I need this fixed immediately")
    assert result.urgent is True


def test_detects_urgency_from_multiple_exclamation_marks():
    result = RuleBasedClassifier().classify("help", "please help me now!!")
    assert result.urgent is True


def test_detects_negative_sentiment():
    result = RuleBasedClassifier().classify("angry", "this is absolutely unacceptable and a scam")
    assert result.negative_sentiment is True


def test_vague_ticket_gets_low_confidence_general_category():
    result = RuleBasedClassifier().classify("hi", "things are kind of weird with my stuff")
    assert result.category == "general"
    assert result.confidence == 0.0


def test_extract_order_id_finds_valid_id():
    assert extract_order_id("please refund ORD-1234 thanks") == "ORD-1234"


def test_extract_order_id_returns_none_when_absent():
    assert extract_order_id("no order number here") is None


def test_extract_order_id_is_case_insensitive():
    assert extract_order_id("issue with ord-5678") == "ORD-5678"
