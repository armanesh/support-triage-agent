"""The router agent: classifies a ticket's category, urgency, and sentiment.

RuleBasedClassifier is the default, and deliberately so: it's fully
deterministic, needs no API key, and its keyword lists are auditable in a
code review in under a minute, which matters for a triage system where a
misclassification routes a ticket to the wrong resolution path. LLMClassifier
is an optional upgrade for handling phrasing the keyword lists don't cover, at
the cost of losing that auditability.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "billing_refund": [
        "refund",
        "charged",
        "charge",
        "billing",
        "invoice",
        "payment",
        "money back",
        "overcharged",
    ],
    "shipping": ["shipping", "delivery", "package", "tracking", "shipped", "arrive", "courier"],
    "account": ["password", "login", "account", "locked", "email address", "username", "2fa"],
    "technical": ["bug", "error", "crash", "broken", "not working", "doesn't work", "glitch"],
}

URGENCY_KEYWORDS = ["urgent", "asap", "immediately", "right now", "emergency"]
NEGATIVE_SENTIMENT_KEYWORDS = [
    "furious",
    "angry",
    "terrible",
    "worst",
    "scam",
    "unacceptable",
    "disgusted",
    "lawyer",
    "sue",
]


@dataclass(frozen=True)
class Classification:
    category: str
    confidence: float
    urgent: bool
    negative_sentiment: bool


class Classifier(ABC):
    @abstractmethod
    def classify(self, subject: str, body: str) -> Classification: ...


class RuleBasedClassifier(Classifier):
    def classify(self, subject: str, body: str) -> Classification:
        text = f"{subject} {body}".lower()

        best_category = "general"
        best_score = 0.0
        for category, keywords in CATEGORY_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text)
            # A real ticket rarely uses more than one or two of a category's
            # keyword synonyms, so dividing by the full list length would
            # unfairly punish a correct single-keyword match. Two matches is
            # treated as fully confident; one match is a reasonable but not
            # certain signal.
            score = min(1.0, matches / 2)
            if score > best_score:
                best_category, best_score = category, score

        urgent = any(kw in text for kw in URGENCY_KEYWORDS) or text.count("!") >= 2
        negative = any(kw in text for kw in NEGATIVE_SENTIMENT_KEYWORDS)

        return Classification(
            category=best_category,
            confidence=best_score,
            urgent=urgent,
            negative_sentiment=negative,
        )


class OpenAIClassifier(Classifier):
    """Optional LLM-backed classifier. Only imports the SDK if selected."""

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model

    def classify(self, subject: str, body: str) -> Classification:
        import json as _json

        categories = list(CATEGORY_KEYWORDS) + ["general"]
        prompt = (
            "Classify this support ticket. Respond with strict JSON only: "
            f'{{"category": one of {categories}, "confidence": 0-1 float, '
            '"urgent": bool, "negative_sentiment": bool}}\n\n'
            f"Subject: {subject}\nBody: {body}"
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = _json.loads(response.choices[0].message.content)
        return Classification(
            category=data["category"],
            confidence=float(data["confidence"]),
            urgent=bool(data["urgent"]),
            negative_sentiment=bool(data["negative_sentiment"]),
        )


def get_classifier(provider: str) -> Classifier:
    if provider == "rule-based":
        return RuleBasedClassifier()
    if provider == "openai":
        return OpenAIClassifier()
    raise ValueError(f"Unknown classifier provider: {provider!r}")


ORDER_ID_RE = re.compile(r"\b(ORD-\d{4,})\b", re.IGNORECASE)


def extract_order_id(text: str) -> str | None:
    match = ORDER_ID_RE.search(text)
    return match.group(1).upper() if match else None
