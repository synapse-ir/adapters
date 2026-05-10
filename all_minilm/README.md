# all-MiniLM-L6-v2 — SYNAPSE Adapter

## What this adapter does

`AllMiniLML6V2Adapter` wraps **sentence-transformers/all-MiniLM-L6-v2**, a lightweight (~80 MB) open-source model that converts text into **384-dimensional dense floating-point vectors** called *embeddings*.

Embeddings capture the semantic meaning of a sentence as a point in a high-dimensional space. Two sentences with similar meaning land close together; semantically unrelated sentences land far apart. This makes embeddings the foundation for:

- **Semantic similarity** — score how alike two texts are without keyword matching
- **Semantic search** — retrieve the most relevant documents for a query
- **Clustering** — group documents by topic without hand-crafted rules
- **Retrieval-augmented generation (RAG)** — find relevant context to feed into an LLM

The adapter follows the SYNAPSE pure-function contract: `ingress` prepares the model input and `egress` converts the numpy output into a `CanonicalIR` with `modality="embedding"`.

---

## License

**Apache 2.0** — fully open source, no license key or commercial agreement required.

---

## Installation

```bash
pip install synapse-adapter-sdk
pip install sentence-transformers
```

The first call to `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")` downloads the model weights (~80 MB) from the Hugging Face Hub and caches them locally. Subsequent calls load from cache.

---

## Architecture: pure functions + caller-owned model

SYNAPSE adapters are **pure transformation functions** — they never load or call the model directly. The caller owns the `SentenceTransformer` instance and is responsible for the actual `model.encode()` call.

```
CanonicalIR
    │
    ▼  adapter.ingress(ir)
{"sentences": ["..."]}          ← dict ready for model.encode()
    │
    ▼  model.encode(input["sentences"])
np.ndarray shape (1, 384)       ← float32 array from sentence-transformers
    │
    ▼  adapter.egress(output, ir, latency_ms)
CanonicalIR                     ← modality="embedding", payload.vector=[...384 floats]
```

This separation keeps the adapter stateless, testable without GPU/network, and reusable across any execution environment.

---

## Complete usage example

```python
import time
from sentence_transformers import SentenceTransformer
from synapse_sdk.types import CanonicalIR, TaskHeader, Payload, ComplianceEnvelope, TaskType, Domain
from all_minilm_adapter import AllMiniLML6V2Adapter
import uuid

# Load the model once (downloads ~80 MB on first run)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
adapter = AllMiniLML6V2Adapter()

# Build a CanonicalIR with the text you want to embed
ir = CanonicalIR(
    ir_version="1.0.0",
    message_id=str(uuid.uuid4()),
    task_header=TaskHeader(
        task_type=TaskType.embed,
        domain=Domain.general,
        priority=2,
        latency_budget_ms=500,
    ),
    payload=Payload(
        modality="text",
        content="Transformers changed the landscape of NLP.",
    ),
    compliance_envelope=ComplianceEnvelope(),
)

# 1. Prepare model input
model_input = adapter.ingress(ir)
# -> {"sentences": ["Transformers changed the landscape of NLP."]}

# 2. Run the model (caller's responsibility)
t0 = time.monotonic()
model_output = model.encode(model_input["sentences"])  # np.ndarray (1, 384)
latency_ms = int((time.monotonic() - t0) * 1000)

# 3. Convert output back to canonical IR
result_ir = adapter.egress(model_output, ir, latency_ms=latency_ms)

# 4. Access the embedding
vector = result_ir.payload.vector        # list[float], length 384
dim    = result_ir.payload.vector_dim    # 384
model_used = result_ir.payload.embedding_model  # "sentence-transformers/all-MiniLM-L6-v2"
confidence  = result_ir.provenance[-1].confidence  # 1.0

print(f"Embedding dim: {dim}")
print(f"First 5 values: {vector[:5]}")
```

---

## Canonical IR mapping

| `model.encode()` output | Canonical IR field | Notes |
|---|---|---|
| `numpy.ndarray` shape `(1, 384)` | `payload.vector` | Converted to `list[float]` via `.tolist()` |
| `384` (fixed constant) | `payload.vector_dim` | Always 384 for this model |
| `"sentence-transformers/all-MiniLM-L6-v2"` | `payload.embedding_model` | Copied from `MODEL_ID` |
| `"embedding"` | `payload.modality` | Set by egress |
| N/A (always valid) | `provenance[-1].confidence` | Fixed at `1.0` — see note below |
| wall-clock ms | `provenance[-1].latency_ms` | Supplied by caller |

---

## Note on confidence

Embedding models do not produce a confidence score — the output is always a complete, valid vector or the model raises an exception. There is no concept of a partially-confident embedding.

`confidence=1.0` is therefore the correct value: it signals "this embedding is fully valid," which is true for every successful `model.encode()` call. It is *not* a measure of semantic certainty about the input text.

---

## Edge-case behaviour

| Situation | egress behaviour |
|---|---|
| `model_output` is wrong shape (e.g. `(1, 128)`) | `vector=[]`, `vector_dim=0` — no raise |
| `model_output` is `None` or a non-array type | `vector=[]`, `vector_dim=0` — no raise |
| `ir.payload.content` is `None` | `ingress` returns `{"sentences": [""]}` |
| Input `modality` is not `"text"` | `ingress` treats missing content as `""` |

---

## Validate

```bash
synapse-validate --adapter all_minilm_adapter.AllMiniLML6V2Adapter --all-fixtures
```

Or via Python:

```bash
uv run python -c "
from synapse_sdk.testing import AdapterValidator
from all_minilm_adapter import AllMiniLML6V2Adapter
AdapterValidator(AllMiniLML6V2Adapter()).assert_valid()
print('All rules passed')
"
```

---

## Links

- [Hugging Face model page](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [sentence-transformers documentation](https://www.sbert.net/)
- [SYNAPSE adapter spec](https://github.com/synapse-ir/spec)
