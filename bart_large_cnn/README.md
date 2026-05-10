# facebook/bart-large-cnn — SYNAPSE Adapter

## What this adapter does

`facebook/bart-large-cnn` generates **abstractive** summaries of English-language text. The model was fine-tuned on the [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail) dataset.

**Abstractive vs extractive summarisation**

| Approach | Behaviour | Example |
|---|---|---|
| **Extractive** | Selects and stitches together sentences from the source | "The company reported record profits. Revenue rose 12%." |
| **Abstractive** | Generates novel sentences that paraphrase the source | "The firm posted its highest-ever earnings, with a 12% revenue gain." |

`bart-large-cnn` is abstractive: the output is a fluent paragraph written by the model, not a copy-paste of source sentences. Summaries may paraphrase, merge, or re-order information.

**Best suited for:** news articles, analyst reports, dense prose with 1–2 central topics, documents up to ~600 words.

**Less suited for:** bullet-point lists, code, tabular data, highly technical domain content (legal/medical jargon), text shorter than ~50 words.

---

## License

**MIT** — fully open source, no license required. The model weights are hosted on Hugging Face and downloaded automatically on first use.

---

## Installation

```bash
pip install synapse-adapter-sdk
pip install transformers torch
```

The model (~1.6 GB) is downloaded and cached by Hugging Face on first use.

---

## IMPORTANT: 1024-token input limit

BART's encoder has a **hard limit of 1024 tokens**, which corresponds to roughly **3 000–4 000 characters** of typical English prose (about 500–700 words).

**What happens to longer text:** the `transformers` pipeline silently truncates input that exceeds 1024 tokens. Only the leading portion of the document is summarised; the rest is discarded without warning.

**The caller is responsible for chunking.** This adapter does **not** split text internally — chunking would require multiple model invocations and belongs in the orchestration layer. If your document exceeds ~3 000 characters, split it into chunks before calling the pipeline and run the adapter once per chunk.

```python
# Rough character limit — adjust based on average token length in your domain
MAX_CHARS = 3_500

chunks = [text[i:i + MAX_CHARS] for i in range(0, len(text), MAX_CHARS)]
summaries = []
for chunk in chunks:
    chunk_ir = ir_with_content(chunk)
    model_input = adapter.ingress(chunk_ir)
    output = summarizer(
        model_input["text"],
        max_length=model_input["max_length"],
        min_length=model_input["min_length"],
        do_sample=model_input["do_sample"],
    )
    result_ir = adapter.egress(output, chunk_ir, latency_ms=0)
    summaries.append(result_ir.payload.content)

final_summary = " ".join(summaries)
```

---

## Architecture: pure functions + caller-owned pipeline

The adapter never loads or invokes the model. The caller owns the `transformers` pipeline instance. The adapter provides two pure, side-effect-free transformation steps:

```
CanonicalIR
    │
    ▼ adapter.ingress(ir)
    │
{"text": "...", "max_length": 130, "min_length": 30, "do_sample": False}
    │
    ▼ summarizer(text, max_length=..., min_length=..., do_sample=...)   ← caller drives this
    │
[{"summary_text": "Generated summary text."}]
    │
    ▼ adapter.egress(model_output, ir, latency_ms)
    │
CanonicalIR  (payload.content = summary, payload.content_length = len(summary))
```

---

## Complete usage example

```python
import time
from transformers import pipeline
from bart_large_cnn_adapter import BartLargeCNNAdapter
from synapse_sdk.types import CanonicalIR, ComplianceEnvelope, Domain, Payload, TaskHeader, TaskType
import uuid

# --- Build a CanonicalIR ---
ir = CanonicalIR(
    ir_version="1.0.0",
    message_id=str(uuid.uuid4()),
    task_header=TaskHeader(
        task_type=TaskType.summarize,
        domain=Domain.general,
        priority=2,
        latency_budget_ms=5000,
    ),
    payload=Payload(
        modality="text",
        content=(
            "The Federal Reserve raised interest rates by 25 basis points on Wednesday, "
            "bringing the benchmark federal-funds rate to a 22-year high. Chair Jerome Powell "
            "said the committee would proceed carefully and that further increases were possible "
            "depending on incoming economic data. Markets initially fell on the announcement but "
            "recovered by the end of the session as investors digested the cautious tone."
        ),
    ),
    compliance_envelope=ComplianceEnvelope(),
)

# --- Load the pipeline once (caller's responsibility) ---
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
adapter    = BartLargeCNNAdapter()

# --- 1. Prepare model input ---
model_input = adapter.ingress(ir)
# {"text": "...", "max_length": 130, "min_length": 30, "do_sample": False}

# --- 2. Run the model ---
t0 = time.monotonic()
model_output = summarizer(
    model_input["text"],
    max_length=model_input["max_length"],
    min_length=model_input["min_length"],
    do_sample=model_input["do_sample"],
)
latency_ms = int((time.monotonic() - t0) * 1000)
# [{"summary_text": "The Fed raised rates to a 22-year high. Chair Powell said ..."}]

# --- 3. Convert output back to canonical IR ---
result_ir = adapter.egress(model_output, ir, latency_ms=latency_ms)

# --- 4. Access the summary ---
print(result_ir.payload.content)         # the generated summary (replaces original)
print(result_ir.payload.content_length)  # character count of the summary
print(result_ir.provenance[-1])          # confidence=1.0, latency_ms=<measured>
```

---

## Canonical IR mapping

### Ingress (CanonicalIR → pipeline input)

| Pipeline parameter | Canonical IR field | Notes |
|---|---|---|
| `text` (positional) | `payload.content` | Empty string `""` when `content` is `None` |
| `max_length` | — | Fixed default: `130` tokens |
| `min_length` | — | Fixed default: `30` tokens |
| `do_sample` | — | Fixed: `False` (deterministic output) |

### Egress (pipeline output → CanonicalIR)

| Pipeline output | Canonical IR field | Notes |
|---|---|---|
| `output[0]["summary_text"]` | `payload.content` | **Replaces** original source text |
| `len(summary_text)` | `payload.content_length` | Character count of the summary |
| — | `payload.entities` | Not set — summarisation produces no entities |
| — | `provenance[-1].confidence` | Fixed at `1.0` (see Note on confidence below) |
| — | `payload.modality` | Always `"text"` |

### Fields carried forward unchanged

| Field | Behaviour |
|---|---|
| `task_header` | Carried forward unchanged |
| `compliance_envelope` | Carried forward unchanged |
| Existing `provenance` entries | Not modified; one new entry appended |

---

## Note on confidence

Provenance `confidence` is fixed at **1.0** for `bart-large-cnn`.

Generative models do not emit a document-level confidence score analogous to a classifier's softmax probability. The model either produces a coherent summary or raises an exception — there is no spectrum of partial quality expressed numerically by the model itself. Setting confidence to `1.0` accurately reflects that any successfully returned summary is a complete, valid output; a failure surfaces as an exception at the pipeline call site, not as a low-confidence result.

---

## Validate

```bash
synapse-validate --adapter bart_large_cnn_adapter.BartLargeCNNAdapter --all-fixtures
```

Or run the test suite directly:

```bash
uv run pytest tests/test_bart_large_cnn_adapter.py -v
```

---

## Links

- [facebook/bart-large-cnn on Hugging Face](https://huggingface.co/facebook/bart-large-cnn)
- [BART paper: Lewis et al., 2019](https://arxiv.org/abs/1910.13461)
- [CNN/DailyMail dataset](https://huggingface.co/datasets/cnn_dailymail)
- [Hugging Face `transformers` summarisation docs](https://huggingface.co/docs/transformers/tasks/summarization)
- [SYNAPSE adapter spec](https://github.com/synapse-ir/spec)
