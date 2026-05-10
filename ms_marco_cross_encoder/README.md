# SYNAPSE Adapter — cross-encoder/ms-marco-MiniLM-L6-v2

## Overview

This adapter wraps the MS MARCO MiniLM cross-encoder model for passage
relevance scoring within the SYNAPSE adapter framework. It handles the `rank`
task type with `general` domain.

Cross-encoders jointly encode a query and a passage in a single forward pass,
enabling fine-grained relevance modelling that bi-encoder retrievers cannot
achieve. The trade-off is cost: every query+passage pair requires its own
forward pass, making cross-encoders ideal as a reranking step after an initial
cheap retrieval stage.

The adapter is fully compliant with the SYNAPSE pure-function contract:
`ingress` and `egress` are stateless transforms. The caller owns the
`CrossEncoder` model instance and drives inference.

---

## Model details

| Field | Value |
|---|---|
| **Model ID** | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| **Architecture** | MiniLM-L6 (12M params, 6 layers) |
| **Task** | Passage relevance scoring / reranking |
| **Fine-tuned on** | MS MARCO Passage Ranking dataset |
| **NDCG@10** | 74.30 on TREC DL 2019 |
| **MRR@10** | 39.01 on MS MARCO Dev |
| **License** | Apache 2.0 |
| **HF page** | https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2 |

---

## Verified output schema

`CrossEncoder.predict()` returns a **raw logit** (unbounded float, not a probability):

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
scores = model.predict([("How many people live in Berlin?", "Berlin had a population...")])
# numpy.ndarray([8.607138], dtype=float32)
# scores[0] is the raw logit — unbounded, can be negative or > 1
```

The egress method applies a **sigmoid transform** to normalise the logit to
[0.0, 1.0]:

```
score = 1 / (1 + exp(-raw_logit))
```

| Raw logit | Sigmoid score | Interpretation |
|---|---|---|
| +∞ | → 1.0 | Perfectly relevant |
| +8.6 | ≈ 0.9998 | Highly relevant |
| 0.0 | = 0.5 | Neutral / uncertain |
| -3.0 | ≈ 0.047 | Not relevant |
| −∞ | → 0.0 | Completely irrelevant |

---

## Usage example

```python
import time
from sentence_transformers import CrossEncoder
from ms_marco_cross_encoder_adapter import MsMarcoCrossEncoderAdapter

# Caller owns the model — load once, reuse
model   = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
adapter = MsMarcoCrossEncoderAdapter()

# 1. Ingress: convert CanonicalIR to model input
model_input = adapter.ingress(ir)
# {"query": "How many people live in Berlin?",
#  "passage": "Berlin had a population of 3.7 million in 2022."}

# 2. Run the model (caller's responsibility)
t0 = time.monotonic()
scores = model.predict([(model_input["query"], model_input["passage"])])
latency_ms = int((time.monotonic() - t0) * 1000)
# numpy.ndarray([8.607138], dtype=float32)

# 3. Egress: convert model output back to CanonicalIR
result_ir = adapter.egress(scores, ir, latency_ms=latency_ms)

# 4. Access the normalised relevance score in [0.0, 1.0]
relevance = result_ir.payload.score
```

The query is read from `ir.task_header.query`; the passage is read from
`ir.payload.content`. Both default to `""` when `None`.

---

## RAG pipeline note

Cross-encoder reranking sits between the retrieval and generation stages:

```
[User Query]
     │
     ▼
[1. Retrieve]   — dense (FAISS / Pinecone) or sparse (BM25) retriever
                   returns top-k candidate passages (k = 50–200)
     │
     ▼
[2. Rerank]     — cross-encoder scores every (query, passage) pair
                   and re-orders by relevance  ← THIS ADAPTER
     │
     ▼
[3. Generate]   — top-n reranked passages (n = 3–10) injected as
                   context into an LLM prompt
```

Because each forward pass is independent, call this adapter in a loop over
candidates. Sort the returned `payload.score` values descending to get the
reranked order. Only the top-n passages are typically passed to the generator.

---

## License

The model is released under the **Apache 2.0** license. The adapter code in
this repository is governed by the SYNAPSE project license.
