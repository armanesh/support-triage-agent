# support-triage-agent

A multi-agent customer support triage system: a router agent classifies each
incoming ticket, a resolver agent picks and calls tools to attempt an answer,
and an escalation agent decides whether that answer is trustworthy enough to
send automatically or should go to a human. Runs fully offline with
deterministic, rule-based agents by default, with an optional LLM-backed
router.

## Why this project

"An agent that calls an LLM in a loop" is not, by itself, a demonstration of
agent design; the part that's actually hard, and actually valuable to show,
is the handoff logic and the judgment call about when *not* to trust the
system's own output. This project is built around that judgment call: every
ticket produces a full trace of what each agent decided and why, and the
escalation policy is a small, readable set of rules rather than a black box.
That's the same shape as a production support triage system, just without
the LLM API bill.

## Architecture

```
Ticket (subject, body, optional order_id)
        │
        ▼
  Router agent (classify.py)
  rule-based keyword classifier: category, urgency, sentiment
        │
        ▼
  Resolver agent (resolver.py)
  picks tools based on category, produces a draft answer + confidence
        │
        ├── KnowledgeBaseTool   (keyword search over policy docs)
        ├── OrderStatusTool     (mock order lookup)
        └── RefundCalculatorTool (pure business-rule function)
        │
        ▼
  Escalation agent (escalation.py)
  overrides confidence entirely for urgency/negative sentiment;
  otherwise escalates below a confidence threshold
        │
   ┌────┴─────┐
   ▼          ▼
Auto-reply   Escalation queue (data/escalations.jsonl)
```

Every run produces a `trace`: a line per agent describing what it decided.
That's what turns "the agent handled it" into something you can actually
review after the fact.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

triage run                     # processes the 10 bundled sample tickets
triage show-queue              # see what got escalated to a human, and why

triage handle "I'd like a refund for ORD-1001" --subject "refund"
triage handle "I am FURIOUS this is a scam!!" --subject "angry"

pytest
```

## What the sample tickets demonstrate

The 10 tickets in `data/sample_tickets.jsonl` aren't random; they're picked to
exercise every branch of the escalation logic:

- A clean refund request with a valid, in-window order: auto-resolved.
- A refund request past the 30-day window: auto-resolved with the 50% rule
  correctly applied.
- A refund request past 60 days: auto-resolved as "not eligible," not escalated,
  since a confident "no" is still a confident answer.
- A shipping status check with a valid order: auto-resolved.
- An angry, "scam"/"unacceptable"-worded ticket: escalated on sentiment,
  regardless of how confident the resolver would have been.
- An urgent, all-caps, double-exclamation-mark ticket: escalated on urgency.
- A vague, low-signal ticket the router can't confidently categorize:
  escalated on low router confidence.
- A technical and an account question the knowledge base actually covers:
  auto-resolved.

## Known limitation, stated plainly

The shipping resolver reports order status but doesn't reason about whether
that status is *itself* a problem (a package stuck for a week should probably
escalate even if the lookup succeeded and the resolver is "confident"). Ticket
T-010 in the sample set exercises exactly this gap on purpose. A production
version would add a rule like "shipping status unchanged for >5 business days
-> escalate" to the resolver, using the same OrderStatusTool data that's
already being fetched. Left out here to keep the resolver's branching logic
readable rather than becoming a rules engine, but it's the natural next
addition.

## Using an LLM-backed router instead of keywords

```bash
pip install -e ".[openai]"
export OPENAI_API_KEY=sk-...
triage run --classifier openai
```

The keyword-based `RuleBasedClassifier` stays the default because its
decisions are auditable in a code review; the LLM classifier is there to show
the same `Classifier` interface supports a strictly more powerful, less
auditable backend without changing the resolver, escalation policy, or
orchestrator at all.

## What a production version would add

- A real ticketing system integration (Zendesk/Freshdesk API) instead of a
  JSONL escalation file.
- The shipping-status escalation rule described above.
- A feedback loop: track which auto-resolved tickets the customer reopened,
  and use that to recalibrate the confidence thresholds in `config.py` over
  time instead of hand-picking them once.

## License

MIT, see `LICENSE`.
