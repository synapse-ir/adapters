# ner_clinical — SYNAPSE Adapter

SYNAPSE adapter for the **John Snow Labs `ner_clinical`** model from Spark NLP for Healthcare.

---

## What this adapter does

`ner_clinical` is a clinical Named Entity Recognition model that identifies medically relevant spans in free text. It was trained on the **i2b2 2010 Clinical NLP Challenge** dataset and uses a Bidirectional LSTM-CNN (NerDL) architecture with pre-trained clinical word embeddings.

The adapter extracts exactly three entity types:

| Label | Description |
|---|---|
| `PROBLEM` | Diagnoses, symptoms, and disease mentions — e.g. *"diabetes"*, *"chest pain"* |
| `TEST` | Lab tests, imaging studies, and diagnostic procedures — e.g. *"CBC"*, *"chest X-ray"* |
| `TREATMENT` | Medications, procedures, and therapeutic interventions — e.g. *"metformin"*, *"bypass surgery"* |

Input is clinical free text (discharge summaries, progress notes, referral letters). Output is a canonical IR populated with `Entity` objects mapped to the three label types above.

---

## License requirement

> **This model requires a valid John Snow Labs Healthcare NLP license.**

`ner_clinical` is part of **Spark NLP for Healthcare**, a commercial product. It is **not open source** and cannot be used without purchasing a license from John Snow Labs.

- Purchase / trial license: **https://www.johnsnowlabs.com/spark-nlp-health/**
- License documentation: https://nlp.johnsnowlabs.com/docs/en/license_getting_started

The adapter code itself is MIT-licensed (see `LICENSE`), but the underlying model will refuse to load at runtime without a valid Healthcare NLP license key set in your environment.

---

## Installation

```bash
# Install the SYNAPSE Adapter SDK
pip install synapse-adapter-sdk

# Install the John Snow Labs Python client (which pulls Spark NLP for Healthcare)
pip install johnsnowlabs

# Authenticate your license (required before any model load)
from johnsnowlabs import nlp
nlp.settings.license = "YOUR_LICENSE_KEY"  # or set JSL_LICENSE env var
```

Apache Spark and a compatible Java runtime (Java 8 or 11) must also be present. See the [Spark NLP installation guide](https://nlp.johnsnowlabs.com/docs/en/install) for platform-specific instructions.

---

## Architecture: how the adapter fits into a Spark NLP pipeline

`NerClinicalAdapter` follows the **SYNAPSE pure-function contract**: `ingress` and `egress` are stateless transforms with no network calls and no Spark session. The Spark NLP pipeline is owned entirely by the caller.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Caller code                                                        │
│                                                                     │
│  1. adapter.ingress(ir)        ← pure function, no I/O             │
│        │                                                            │
│        ▼  {"text": "..."}                                           │
│  2. spark_pipeline.transform(spark_df)   ← caller owns Spark       │
│        │                                                            │
│        ▼  ner_chunk annotations (list of Annotation objects)       │
│  3. adapter.egress(ner_chunks, ir, latency_ms)  ← pure function    │
│        │                                                            │
│        ▼  CanonicalIR with payload.entities populated              │
└─────────────────────────────────────────────────────────────────────┘
```

The required Spark NLP pipeline stages (in order):

1. `DocumentAssembler` — `text` → `document`
2. `SentenceDetector` — `document` → `sentence`
3. `Tokenizer` — `sentence` → `token`
4. `WordEmbeddingsModel` — `embeddings_clinical` (en, clinical/models)
5. `MedicalNerModel` — `ner_clinical` (en, clinical/models)
6. `NerConverter` — produces `ner_chunk`

---

## Complete usage example

```python
import time
import pandas as pd
from pyspark.sql import SparkSession
from johnsnowlabs import nlp
from sparknlp.annotator import (
    DocumentAssembler, SentenceDetector, Tokenizer, NerConverter
)
from sparknlp_jsl.annotator import MedicalNerModel, WordEmbeddingsModel
from pyspark.ml import Pipeline

from synapse_sdk.types import CanonicalIR, TaskHeader, Payload, TaskType, Domain
from ner_clinical_adapter import NerClinicalAdapter
import uuid

# ── 1. Set up Spark session (caller's responsibility) ──────────────────
spark = SparkSession.builder \
    .appName("ner_clinical_synapse") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp-jsl_2.12:5.x.x") \
    .getOrCreate()

# ── 2. Build the Spark NLP pipeline (caller's responsibility) ──────────
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCols(["sentence"])

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner_model = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[
    document_assembler, sentence_detector, tokenizer,
    embeddings, ner_model, ner_converter,
])
fitted_pipeline = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

# ── 3. Build a canonical IR ────────────────────────────────────────────
ir = CanonicalIR(
    ir_version="1.0.0",
    message_id=str(uuid.uuid4()),
    task_header=TaskHeader(
        task_type=TaskType.extract,
        domain=Domain.medical,
        priority=2,
        latency_budget_ms=5000,
    ),
    payload=Payload(
        modality="text",
        content="The patient was diagnosed with type 2 diabetes. "
                "HbA1c was ordered. Metformin 500mg was prescribed.",
    ),
)

# ── 4. Run the adapter ─────────────────────────────────────────────────
adapter = NerClinicalAdapter()

# ingress: extract the text for the pipeline
model_input = adapter.ingress(ir)          # {"text": "The patient was diagnosed..."}

# run the Spark pipeline
df = spark.createDataFrame(pd.DataFrame([model_input]))
t0 = time.perf_counter()
result_df = fitted_pipeline.transform(df)
latency_ms = int((time.perf_counter() - t0) * 1000)

# collect ner_chunk annotations
ner_chunks = result_df.select("ner_chunk").first()["ner_chunk"]

# egress: convert to canonical IR
result_ir = adapter.egress(ner_chunks, ir, latency_ms=latency_ms)

# ── 5. Inspect results ─────────────────────────────────────────────────
for entity in result_ir.payload.entities:
    print(f"[{entity.label:10s}] {entity.text!r:30s}  conf={entity.confidence:.4f}  "
          f"offset={entity.start}:{entity.end}")

print(f"\nProvenance: model={result_ir.provenance[-1].model_id}  "
      f"confidence={result_ir.provenance[-1].confidence:.4f}  "
      f"latency={result_ir.provenance[-1].latency_ms}ms")
```

Expected output (approximate):
```
[PROBLEM   ] 'type 2 diabetes'              conf=0.9823  offset=30:45
[TEST      ] 'HbA1c'                        conf=0.9671  offset=48:52
[TREATMENT ] 'Metformin 500mg'              conf=0.9788  offset=69:83

Provenance: model=johnsnowlabs/ner_clinical  confidence=0.9761  latency=1240ms
```

---

## Entity types

| Label | Maps to | Typical spans |
|---|---|---|
| `PROBLEM` | Diagnoses, conditions, symptoms | *"diabetes mellitus"*, *"chest pain"*, *"shortness of breath"* |
| `TEST` | Diagnostic tests and procedures | *"HbA1c"*, *"chest X-ray"*, *"complete blood count"*, *"biopsy"* |
| `TREATMENT` | Medications, surgeries, therapies | *"metformin"*, *"insulin"*, *"coronary bypass"*, *"chemotherapy"* |

These are the **only** three labels this model produces. No other labels will appear.

---

## Canonical IR mapping

The `NerConverter` `ner_chunk` output is mapped to canonical IR `Entity` fields as follows:

| Spark NLP `ner_chunk` field | Canonical `Entity` field | Notes |
|---|---|---|
| `annotation.result` | `entity.text` | Surface text of the extracted span |
| `annotation.metadata["entity"]` | `entity.label` | One of `PROBLEM`, `TEST`, `TREATMENT` |
| `annotation.begin` | `entity.start` | Character start offset (inclusive) |
| `annotation.end` | `entity.end` | Character end offset (inclusive) |
| `annotation.metadata["confidence"]` | `entity.confidence` | Cast from `str` → `float`, clamped to `[0.0, 1.0]` |

**Important:** `ner_chunk` stores `confidence` as a **string** (e.g. `"0.9876"`), not a float. The adapter casts it with `float()` before storing it. Malformed strings default to `0.0`.

The `ProvenanceEntry.confidence` attached to the output IR is the **mean** of all extracted entity confidences. If no entities were extracted it is `0.0`.

---

## PII handling (G-S04)

`PROBLEM` entities can contain patient-identifying information (e.g. rare disease names combined with demographic context). The adapter implements conservative G-S04 PII propagation:

1. **Incoming flag propagation** — if the input IR's `compliance_envelope.pii_present` is `True`, the output IR preserves `pii_present=True` regardless of what entities were extracted.
2. **Conservative upgrade** — if any `PROBLEM` entity is extracted, the adapter upgrades `pii_present` to `True` even if the incoming IR had `pii_present=None` or `False`.
3. **No downgrade** — the adapter never sets `pii_present` to `False` or `None`.
4. **TEST / TREATMENT entities** — do not trigger the PII upgrade. Only `PROBLEM` does.

All other `compliance_envelope` fields (`required_tags`, `data_residency`, `retention_policy`, `purpose_limitation`) are carried forward unchanged.

---

## Validate

Run the full 13-rule §2.4 validation suite against all 20 standard fixtures:

```bash
# From the adapters/ directory
uv run synapse-validate --adapter ner_clinical_adapter.NerClinicalAdapter --all-fixtures
```

Or run just the test suite:

```bash
uv run pytest tests/test_ner_clinical_adapter.py -v
```

---

## Links

- **SYNAPSE spec**: https://github.com/synapse-ir/spec
- **John Snow Labs**: https://www.johnsnowlabs.com
- **ner_clinical on NLP Models Hub**: https://nlp.johnsnowlabs.com/2021/03/31/ner_clinical_en.html
- **Spark NLP for Healthcare license**: https://www.johnsnowlabs.com/spark-nlp-health/
- **i2b2 2010 dataset**: https://www.i2b2.org/NLP/DataSets/Main.php
