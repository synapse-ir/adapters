# bart-large-mnli — SYNAPSE Adapter

## Overview

Zero-shot text classification via Natural Language Inference (NLI). The model frames every classification as an entailment task: for each candidate label it checks whether the input text entails "This example is about \<label\>." The softmax over entailment logits produces a ranked probability distribution across any labels the caller supplies at inference time — no fine-tuning or fixed vocabulary required.

## Model details

| Field | Value |
|-------|-------|
| Model ID | `facebook/bart-large-mnli` |
| Architecture | BART-large fine-tuned on MultiNLI |
| Task | Zero-shot classification |
| Domain | General |
| License | MIT |
| Downloads | ~3.5 million/month (Hugging Face) |
| Training data | MultiNLI (Multi-Genre Natural Language Inference Corpus) |

## Verified output schema

The `pipeline("zero-shot-classification")` call returns a single dict (not a list):

```python
{
    "sequence": "one day I will see the world",
    "labels":   ["travel", "dancing", "cooking"],   # sorted descending by score
    "scores":   [0.9938651323318481, 0.0032737774308770895, 0.002861034357920289]
}
```

- `sequence` — the original input text
- `labels` — all candidate labels, sorted by descending score; `labels[0]` is always the top prediction
- `scores` — softmax probabilities in the same order as `labels`; sum to ~1.0
- Confidence = `scores[0]` (probability of the top predicted label)

## Usage example

```python
from transformers import pipeline
from bart_large_mnli_adapter import BartLargeMnliAdapter
from synapse_sdk.types import CanonicalIR, TaskHeader, TaskType, Domain, Payload

# Build a CanonicalIR with candidate labels in task_header
ir = CanonicalIR(
    ir_version="1.0.0",
    message_id="...",
    task_header=TaskHeader(
        task_type=TaskType.classify,
        domain=Domain.general,
        priority=2,
        latency_budget_ms=5000,
        candidate_labels=["travel", "cooking", "dancing"],
    ),
    payload=Payload(modality="text", content="one day I will see the world"),
)

pipe    = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
adapter = BartLargeMnliAdapter()

model_input  = adapter.ingress(ir)
model_output = pipe(model_input["text"], candidate_labels=model_input["candidate_labels"])
result_ir    = adapter.egress(model_output, ir, latency_ms=latency_ms)

top_label = result_ir.payload.labels[0].label   # e.g. "travel"
top_score = result_ir.payload.labels[0].score   # e.g. 0.9938
all_labels = result_ir.payload.labels           # full ranked list
```

## Zero-shot note

The key advantage of zero-shot classification is that **no training data is needed for new label sets**. You can pass any descriptive strings as candidate labels and the model will produce meaningful rankings based on the NLI entailment signal learned from MultiNLI. This makes the adapter suitable for:

- Rapid prototyping without labelled datasets
- Dynamic label sets that change at runtime
- Multi-domain classification with a single model

If `task_header.candidate_labels` is `None` or empty, the adapter defaults to `["positive", "negative", "neutral"]` for general sentiment classification.

## License

MIT. The model weights and this adapter are both MIT licensed. See the [Hugging Face model card](https://huggingface.co/facebook/bart-large-mnli) for details.
