# FinbertAdapter — SYNAPSE Adapter for ProsusAI/finbert

## Overview

FinbertAdapter wraps [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) for use inside
SYNAPSE pipelines. FinBERT classifies financial text as **positive**, **negative**, or
**neutral** — reflecting the market sentiment expressed in earnings calls, news articles,
analyst reports, and similar finance-domain prose.

The adapter exposes two pure transformation steps (`ingress` and `egress`) so that the caller
owns the `transformers` pipeline and all model I/O, while the adapter handles only the
CanonicalIR ↔ model format conversion.

## Model Details

| Field          | Value                                                      |
|----------------|------------------------------------------------------------|
| Model ID       | `ProsusAI/finbert`                                         |
| HuggingFace    | https://huggingface.co/ProsusAI/finbert                    |
| Task           | `text-classification` (financial sentiment)                |
| Domain         | Finance                                                    |
| Output labels  | `positive`, `negative`, `neutral`                          |
| License        | Apache 2.0                                                 |
| Base model     | BERT (`bert-base-uncased`)                                 |
| Training data  | Financial PhraseBank + earnings call transcripts           |

Install dependencies:

```bash
pip install transformers torch
```

## Verified Output Schema

The `transformers` pipeline returns a top-1 list (default behaviour):

```python
from transformers import pipeline

pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")
result = pipe("Revenues increased significantly this quarter.")
# [{'label': 'positive', 'score': 0.9723}]
```

Schema of `result`:

| Field   | Type    | Values                               |
|---------|---------|--------------------------------------|
| `label` | `str`   | `"positive"`, `"negative"`, `"neutral"` (always lowercase) |
| `score` | `float` | Softmax probability of the top class, in `[0.0, 1.0]`      |

The list always contains exactly one element when called with default settings.

## Usage Example

```python
import time
from transformers import pipeline
from finbert_adapter import FinbertAdapter
from synapse_sdk.types import CanonicalIR, Payload, TaskHeader, TaskType, Domain
import uuid

# Build a canonical IR
ir = CanonicalIR(
    ir_version="1.0.0",
    message_id=str(uuid.uuid4()),
    task_header=TaskHeader(
        task_type=TaskType.classify,
        domain=Domain.finance,
        priority=2,
        latency_budget_ms=2000,
    ),
    payload=Payload(
        modality="text",
        content="The company reported record-breaking revenue this quarter.",
    ),
)

# Set up adapter and model (caller owns the pipeline)
adapter = FinbertAdapter()
pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# 1. Prepare model input
model_input = adapter.ingress(ir)
# {"text": "The company reported record-breaking revenue this quarter."}

# 2. Run the model (caller's responsibility)
t0 = time.monotonic()
model_output = pipe(model_input["text"])
latency_ms = int((time.monotonic() - t0) * 1000)
# [{"label": "positive", "score": 0.9712}]

# 3. Convert output back to canonical IR
result_ir = adapter.egress(model_output, ir, latency_ms=latency_ms)

# 4. Access results
label = result_ir.payload.labels[0].label   # "positive"
score = result_ir.payload.labels[0].score   # 0.9712
prov  = result_ir.provenance[-1]
print(f"Sentiment: {label} (confidence={score:.4f}, latency={prov.latency_ms}ms)")
```

## Finance Domain Note

FinBERT labels reflect **market sentiment** expressed in the text, not the emotional tone
of the author:

- **`positive`** — the text conveys optimism, growth, outperformance, or beats expectations.
  Example: *"Revenue grew 18% year-over-year, exceeding analyst forecasts."*

- **`negative`** — the text conveys pessimism, decline, loss, or missed expectations.
  Example: *"The company issued a profit warning and cut its full-year guidance."*

- **`neutral`** — the text is factual, non-directional, or describes routine events.
  Example: *"The board meeting is scheduled for 14 March."*

The `score` is the softmax probability of the top class. A high score (e.g. `0.97`) means
the model is confident; a low score (e.g. `0.42`) suggests the text is ambiguous or
close to a decision boundary.

FinBERT does **not** extract person names or other personally identifiable information.
`compliance_envelope.pii_present` is never upgraded to `True` by this adapter.

## License

Apache 2.0 — see [github.com/ProsusAI/finBERT](https://github.com/ProsusAI/finBERT) for
the full license text.

The SYNAPSE adapter code in this repository is also Apache 2.0.
