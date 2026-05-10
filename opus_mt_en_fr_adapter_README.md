# SYNAPSE Adapter — Helsinki-NLP/opus-mt-en-fr

## Overview

This adapter wraps the Helsinki-NLP Opus-MT English-to-French neural machine
translation model within the SYNAPSE adapter framework. It handles the
`translate` task type with `multilingual` domain.

The adapter follows the SYNAPSE pure-function contract: `ingress` and `egress`
are stateless transformations. The caller owns the `transformers` pipeline
instance and drives inference. The adapter only converts data formats.

---

## Model details

| Field | Value |
|---|---|
| **Model ID** | `Helsinki-NLP/opus-mt-en-fr` |
| **Architecture** | MarianMT (seq2seq Transformer) |
| **Task** | Neural machine translation (English → French) |
| **Training data** | OPUS parallel corpora |
| **License** | Apache 2.0 |
| **Downloads** | ~383 000/month (Hugging Face Hub) |
| **HF page** | https://huggingface.co/Helsinki-NLP/opus-mt-en-fr |

---

## Verified output schema

The `transformers` translation pipeline returns:

```python
from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
result = translator("How are you?")
# [{'translation_text': 'Comment allez-vous ?'}]
```

The output is **always a list with exactly one dict**. The only key is
`"translation_text"` whose value is a plain string (the translated text).
Seq2seq generative models do not emit a token-level confidence score.

---

## Usage example

```python
import time
from transformers import pipeline
from opus_mt_en_fr_adapter import OpusMtEnFrAdapter

# Caller owns the model — load once, reuse
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
adapter    = OpusMtEnFrAdapter()

# 1. Ingress: convert CanonicalIR to model input
model_input = adapter.ingress(ir)
# {"text": "How are you?"}

# 2. Run the model (caller's responsibility)
t0 = time.monotonic()
model_output = translator(model_input["text"])
latency_ms = int((time.monotonic() - t0) * 1000)
# [{"translation_text": "Comment allez-vous ?"}]

# 3. Egress: convert model output back to CanonicalIR
result_ir = adapter.egress(model_output, ir, latency_ms=latency_ms)

# 4. Access the result — payload.content now holds the French translation
french_text = result_ir.payload.content
```

`payload.content` is **replaced** by the translated text. The original English
source is not preserved in the returned IR. Store it separately if needed.

---

## Pattern generality — all Helsinki-NLP opus-mt models

**This adapter pattern works for ALL `Helsinki-NLP/opus-mt-{src}-{tgt}` models**
(1000+ language pairs). The `transformers` translation pipeline produces the
same `[{"translation_text": str}]` output schema across the entire opus-mt
family because all models share the MarianMT architecture.

To use a different language pair, copy this file and change only:

```python
MODEL_ID = "Helsinki-NLP/opus-mt-{src}-{tgt}"
```

The `ingress` and `egress` logic is **identical** for all pairs.

### Example model IDs

| Model ID | Direction |
|---|---|
| `Helsinki-NLP/opus-mt-en-fr` | English → French (this adapter) |
| `Helsinki-NLP/opus-mt-en-de` | English → German |
| `Helsinki-NLP/opus-mt-en-es` | English → Spanish |
| `Helsinki-NLP/opus-mt-zh-en` | Chinese → English |
| `Helsinki-NLP/opus-mt-fr-en` | French → English |
| `Helsinki-NLP/opus-mt-en-ru` | English → Russian |

---

## License

The model is released under the **Apache 2.0** license. The adapter code in
this repository is governed by the SYNAPSE project license.
