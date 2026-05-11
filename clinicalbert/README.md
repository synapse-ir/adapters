# ClinicalBertAdapter - SYNAPSE Adapter for medicalai/ClinicalBERT

## Overview

ClinicalBertAdapter wraps [medicalai/ClinicalBERT](https://huggingface.co/medicalai/ClinicalBERT)
for SYNAPSE pipelines. The model is used through the Hugging Face `fill-mask`
pipeline for masked language modeling over clinical and medical text.

The adapter is intentionally pure: it performs no model loading, no network
calls, and no persistent state changes. The caller owns the `transformers`
pipeline and all inference.

## Model Details

| Field | Value |
| --- | --- |
| Model ID | `medicalai/ClinicalBERT` |
| Hugging Face | https://huggingface.co/medicalai/ClinicalBERT |
| Task | `fill-mask` / masked language modeling |
| Domain | Clinical and medical text |
| Output | Ranked mask replacement candidates |

## Usage

```python
import time

from clinicalbert_adapter import ClinicalBertAdapter
from transformers import pipeline

adapter = ClinicalBertAdapter()
pipe = pipeline("fill-mask", model="medicalai/ClinicalBERT")

model_input = adapter.ingress(ir)

t0 = time.monotonic()
model_output = pipe(model_input["text"])
latency_ms = int((time.monotonic() - t0) * 1000)

result_ir = adapter.egress(model_output, ir, latency_ms=latency_ms)
```

## Ingress

`ingress(ir)` returns:

```python
{"text": ir.payload.content or ""}
```

The adapter preserves `[MASK]` tokens exactly and does not tokenize the text.

## Egress

The verified `transformers` fill-mask output shape is a list of candidates:

```python
[
    {
        "score": 0.8497790098190308,
        "sequence": "the patient reports chest pain after exercise.",
        "token": 38576,
        "token_str": "pain",
    }
]
```

Each valid candidate becomes a `Classification`:

```python
Classification(label="pain", score=0.8497790098190308)
```

Candidates are kept in model order. Empty or malformed outputs produce
`payload.labels == []` and provenance confidence `0.0`. Provenance confidence
equals the first valid candidate score.

## PHI and PII

Clinical text may contain PHI or PII. This adapter does not inspect clinical
content, does not extract entities, and does not set
`compliance_envelope.pii_present = True`. De-identification and PHI handling are
the caller's responsibility. Existing compliance metadata is preserved exactly.

## License and Citation

Review the upstream model card for current license and citation details:
https://huggingface.co/medicalai/ClinicalBERT
