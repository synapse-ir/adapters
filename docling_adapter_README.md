# SYNAPSE Adapter — docling-project/docling

Bridges [Docling](https://github.com/docling-project/docling)'s `DoclingDocument`
output into the SYNAPSE canonical IR, so any downstream SYNAPSE-registered model
can consume a Docling-processed document without writing custom connector code.

## What is Docling?

Docling is an open-source document AI library developed by IBM Research. It converts
PDF, DOCX, PPTX, HTML, AsciiDoc, and Markdown files into a structured
`DoclingDocument` object with typed text blocks (titles, paragraphs, section headers,
lists, footnotes, captions, code, and more), extracted tables, embedded pictures,
and full layout provenance. Docling powers IBM's document understanding pipeline and
is MIT licensed.

## What this adapter does

| Stage | Input | Output |
|-------|-------|--------|
| **ingress** | `CanonicalIR` (payload.content = file path or URL) | `{"source": "<path or url>"}` |
| **egress** | `DoclingDocument` instance (or `export_to_dict()` dict) + original IR | `CanonicalIR` with extracted content |

### Egress mapping

| Docling field | Canonical IR field |
|---|---|
| `doc.export_to_markdown()` | `payload.content` |
| `doc.texts[i].text` | `payload.entities[i].text` |
| `doc.texts[i].label.value` | `payload.entities[i].label` |
| `len(doc.tables)` | `payload.data["docling_table_count"]` |
| `len(doc.pages)` | `payload.data["docling_page_count"]` |
| *(no doc-level score)* | provenance `confidence = 1.0` |

Entity `start`, `end` are `None` — Docling uses bounding boxes, not character
offsets. Entity `confidence` is always `1.0` (Docling emits no per-item score;
`1.0` is the SYNAPSE canonical no-confidence sentinel).

## Installation

Install Docling alongside the SYNAPSE SDK:

```bash
pip install synapse-adapter-sdk docling
```

## End-to-end usage

```python
import time
from docling.document_converter import DocumentConverter
from synapse_sdk.types import CanonicalIR, TaskHeader, Payload, TaskType, Domain
from docling_adapter import DoclingAdapter
import uuid

# 1. Build a canonical IR pointing at the source document.
ir = CanonicalIR(
    ir_version="1.0.0",
    message_id=str(uuid.uuid4()),
    task_header=TaskHeader(
        task_type=TaskType.extract,
        domain=Domain.document,
        priority=1,
        latency_budget_ms=30_000,
    ),
    payload=Payload(
        modality="text",
        content="/path/to/report.pdf",   # or a URL
    ),
)

# 2. Run ingress to obtain the Docling source argument.
adapter = DoclingAdapter()
model_input = adapter.ingress(ir)

# 3. Run Docling's DocumentConverter.
t0 = time.monotonic()
result = DocumentConverter().convert(model_input["source"])
latency_ms = int((time.monotonic() - t0) * 1000)

# 4. Run egress to produce the enriched canonical IR.
result_ir = adapter.egress(result.document, ir, latency_ms)

# result_ir.payload.content  → full markdown representation of the document
# result_ir.payload.entities → list of typed text blocks (title, paragraph, …)
# result_ir.payload.data     → {"docling_table_count": N, "docling_page_count": M}
# result_ir.provenance[-1]   → latency, model_id, confidence
```

### Passing a quality threshold

```python
from synapse_sdk.types import TaskHeader, TaskType, Domain

header = TaskHeader(
    task_type=TaskType.extract,
    domain=Domain.document,
    priority=1,
    latency_budget_ms=30_000,
    quality_floor=0.80,   # forwarded as model_input["quality_floor"]
)
```

### Using a pre-serialized dict

If you have the result of `doc.export_to_dict()` (e.g. stored in a cache),
pass it directly — the adapter handles both forms:

```python
doc_dict = result.document.export_to_dict()
result_ir = adapter.egress(doc_dict, ir, latency_ms)
```

## Entity labels

Labels come directly from Docling's `DocItemLabel` enum. Common values:

`title` · `section_header` · `paragraph` · `list_item` · `caption` ·
`footnote` · `page_header` · `page_footer` · `code` · `formula` · `reference`

## License

This adapter is **MIT** licensed. Docling itself is also MIT licensed. No
proprietary license is required to use either the adapter or the underlying model.
