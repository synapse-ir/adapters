# SYNAPSE Adapter — cardiffnlp/twitter-roberta-base-sentiment-latest

## Overview

This adapter wraps the Cardiff NLP Twitter-RoBERTa sentiment model within the
SYNAPSE adapter framework. It handles the `classify` task type with the
`conversational` domain.

The model is optimised for **social media text** — short, informal posts with
hashtags, mentions, URLs, and colloquial language. It outperforms generic
sentiment models on Twitter and similar platforms because it was pre-trained
and fine-tuned directly on tweet corpora.

The adapter follows the SYNAPSE pure-function contract: `ingress` and `egress`
are stateless transforms. The caller owns the `transformers` pipeline instance
and drives inference.

---

## Model details

| Field | Value |
|---|---|
| **Model ID** | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| **Architecture** | RoBERTa-base (125M params) |
| **Task** | Three-class sentiment classification |
| **Labels** | `Negative`, `Neutral`, `Positive` (title-cased) |
| **Training data** | 124 million tweets, January 2018 – December 2021 |
| **License** | CC BY 4.0 (attribution required) |
| **HF page** | https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest |

---

## Verified output schema

The `transformers` sentiment-analysis pipeline returns:

```python
from transformers import pipeline

pipe = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
)
result = pipe("Covid cases are increasing fast!")
# [{'label': 'Negative', 'score': 0.7236}]
```

The output is **always a list with exactly one dict**. The keys are `"label"`
(a title-cased string) and `"score"` (a softmax probability in [0.0, 1.0]).

---

## Usage example

```python
import time
from transformers import pipeline
from twitter_roberta_sentiment_adapter import TwitterRobertaSentimentAdapter

# Caller owns the model — load once, reuse
pipe = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
)
adapter = TwitterRobertaSentimentAdapter()

# 1. Ingress: convert CanonicalIR to model input
model_input = adapter.ingress(ir)
# {"text": "Covid cases are increasing fast!"}

# 2. Run the model (caller's responsibility)
t0 = time.monotonic()
model_output = pipe(model_input["text"])
latency_ms = int((time.monotonic() - t0) * 1000)
# [{"label": "Negative", "score": 0.7236}]

# 3. Egress: convert model output back to CanonicalIR
result_ir = adapter.egress(model_output, ir, latency_ms=latency_ms)

# 4. Access the result
label = result_ir.payload.labels[0].label  # 'Negative' | 'Neutral' | 'Positive'
score = result_ir.payload.labels[0].score  # float in [0.0, 1.0]
```

---

## Label note — title-cased labels and social media meaning

Labels are returned in **title case**: `'Negative'`, `'Neutral'`, `'Positive'`.
This differs from some other sentiment adapters (e.g. FinBERT) that use
lowercase labels. Callers that route across multiple sentiment adapters must
normalise labels before comparing.

| Label | Meaning in social media context |
|---|---|
| `Positive` | Post expresses approval, excitement, praise, or support |
| `Neutral` | Post is factual, informational, or indifferent |
| `Negative` | Post expresses criticism, frustration, sadness, or opposition |

The model handles Twitter-specific features such as `@mentions`, `#hashtags`,
URLs (replace with `http`), and mixed-case slang. It degrades on formal or
domain-specific prose (legal, medical, financial text).

---

## License

The model weights are released under **CC BY 4.0**. Attribution to the Cardiff
NLP group is required when redistributing or building products based on this
model. This is more restrictive than the MIT or Apache 2.0 licenses used by
other adapters in this collection — check your deployment's licence obligations
before using in a commercial product.

See the full licence: https://creativecommons.org/licenses/by/4.0/
