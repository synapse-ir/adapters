"""Tests for DoclingAdapter — unit behaviour and AdapterValidator suite."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest

from docling_adapter import DoclingAdapter
from synapse_sdk.testing.fixtures import ALL_FIXTURES, GENERAL_EMBED_LARGE, LEGAL_EXTRACT_BASIC
from synapse_sdk.types import (
    CanonicalIR,
    ComplianceEnvelope,
    Domain,
    Payload,
    TaskHeader,
    TaskType,
)
from synapse_sdk.validator import AdapterValidator


# ---------------------------------------------------------------------------
# IR factory helpers
# ---------------------------------------------------------------------------

def _make_ir(
    content: str | None = "/path/to/document.pdf",
    modality: str = "text",
    quality_floor: float | None = None,
    compliance: ComplianceEnvelope | None = None,
) -> CanonicalIR:
    payload_kwargs: dict[str, Any] = {"modality": modality}
    if modality == "text":
        payload_kwargs["content"] = content
    elif modality == "structured":
        payload_kwargs["data"] = {"key": "value"}
    elif modality == "embedding":
        payload_kwargs["vector"] = [0.1, 0.2, 0.3]
        payload_kwargs["vector_dim"] = 3

    return CanonicalIR(
        ir_version="1.0.0",
        message_id=str(uuid.uuid4()),
        task_header=TaskHeader(
            task_type=TaskType.extract,
            domain=Domain.general,
            priority=1,
            latency_budget_ms=5000,
            quality_floor=quality_floor,
        ),
        payload=Payload(**payload_kwargs),
        compliance_envelope=compliance or ComplianceEnvelope(),
    )


def _make_doc_mock(
    texts: list[dict[str, str]] | None = None,
    tables: list[Any] | None = None,
    pages: dict[Any, Any] | None = None,
    markdown: str = "# Title\n\nBody text.",
    markdown_raises: bool = False,
    text_raises: bool = False,
) -> MagicMock:
    """Build a mock DoclingDocument with controlled field values."""
    doc = MagicMock()

    if markdown_raises:
        doc.export_to_markdown.side_effect = RuntimeError("markdown conversion failed")
    else:
        doc.export_to_markdown.return_value = markdown

    if text_raises:
        doc.export_to_text.side_effect = RuntimeError("text conversion failed")
    else:
        doc.export_to_text.return_value = "Fallback plain text."

    mock_texts: list[MagicMock] = []
    for t in texts or []:
        item = MagicMock()
        item.text = t["text"]
        label = MagicMock()
        label.value = t["label"]
        item.label = label
        mock_texts.append(item)

    doc.texts = mock_texts
    doc.tables = tables if tables is not None else []
    doc.pages = pages if pages is not None else {}
    return doc


# ---------------------------------------------------------------------------
# ingress
# ---------------------------------------------------------------------------

def test_ingress_file_path() -> None:
    ir = _make_ir(content="/docs/annual_report.pdf")
    result = DoclingAdapter().ingress(ir)
    assert result == {"source": "/docs/annual_report.pdf"}


def test_ingress_url() -> None:
    ir = _make_ir(content="https://example.com/whitepaper.pdf")
    result = DoclingAdapter().ingress(ir)
    assert result == {"source": "https://example.com/whitepaper.pdf"}


def test_ingress_none_content_returns_empty_source() -> None:
    # Use structured modality to obtain a payload where content is None.
    ir = _make_ir(modality="structured")
    result = DoclingAdapter().ingress(ir)
    assert result == {"source": ""}


def test_ingress_quality_floor_forwarded() -> None:
    ir = _make_ir(content="/docs/report.pdf", quality_floor=0.75)
    result = DoclingAdapter().ingress(ir)
    assert result["source"] == "/docs/report.pdf"
    assert result["quality_floor"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# egress — DoclingDocument instance path
# ---------------------------------------------------------------------------

def test_egress_docling_instance_sets_markdown_as_content() -> None:
    ir = _make_ir()
    doc = _make_doc_mock(markdown="# Report\n\nThis is the body.", texts=[])
    out = DoclingAdapter().egress(doc, ir, latency_ms=100)
    assert out.payload.content == "# Report\n\nThis is the body."


def test_egress_dict_input_handled_gracefully() -> None:
    ir = _make_ir()
    dict_input: dict[str, Any] = {
        "texts": [{"text": "Hello world", "label": "paragraph"}],
        "tables": [],
        "pages": {},
    }
    out = DoclingAdapter().egress(dict_input, ir, latency_ms=50)
    assert out is not None
    assert len(out.provenance) == len(ir.provenance) + 1


def test_egress_texts_entities_count_matches() -> None:
    ir = _make_ir()
    doc = _make_doc_mock(texts=[
        {"text": "Introduction", "label": "section_header"},
        {"text": "Body paragraph text.", "label": "paragraph"},
        {"text": "A footnote reference.", "label": "footnote"},
    ])
    out = DoclingAdapter().egress(doc, ir, latency_ms=80)
    assert out.payload.entities is not None
    assert len(out.payload.entities) == 3


def test_egress_entity_labels_match_label_value() -> None:
    ir = _make_ir()
    doc = _make_doc_mock(texts=[
        {"text": "Section One", "label": "section_header"},
        {"text": "Paragraph content.", "label": "paragraph"},
    ])
    out = DoclingAdapter().egress(doc, ir, latency_ms=60)
    assert out.payload.entities is not None
    labels = [e.label for e in out.payload.entities]
    assert labels == ["section_header", "paragraph"]


def test_egress_entity_confidence_is_1_0() -> None:
    ir = _make_ir()
    doc = _make_doc_mock(texts=[
        {"text": "Title text", "label": "title"},
        {"text": "Body text", "label": "paragraph"},
    ])
    out = DoclingAdapter().egress(doc, ir, latency_ms=70)
    assert out.payload.entities is not None
    assert all(e.confidence == 1.0 for e in out.payload.entities)


def test_egress_entity_start_and_end_are_none() -> None:
    ir = _make_ir()
    doc = _make_doc_mock(texts=[{"text": "Some content", "label": "paragraph"}])
    out = DoclingAdapter().egress(doc, ir, latency_ms=50)
    assert out.payload.entities is not None
    assert all(e.start is None for e in out.payload.entities)
    assert all(e.end is None for e in out.payload.entities)


# ---------------------------------------------------------------------------
# egress — metadata counts
# ---------------------------------------------------------------------------

def test_egress_tables_present_sets_docling_table_count() -> None:
    ir = _make_ir()
    doc = _make_doc_mock(tables=[MagicMock(), MagicMock()], texts=[])
    out = DoclingAdapter().egress(doc, ir, latency_ms=50)
    assert out.payload.data is not None
    assert out.payload.data["docling_table_count"] == 2


def test_egress_pages_present_sets_docling_page_count() -> None:
    ir = _make_ir()
    doc = _make_doc_mock(pages={1: MagicMock(), 2: MagicMock(), 3: MagicMock()}, texts=[])
    out = DoclingAdapter().egress(doc, ir, latency_ms=50)
    assert out.payload.data is not None
    assert out.payload.data["docling_page_count"] == 3


def test_egress_no_tables_omits_docling_table_count() -> None:
    ir = _make_ir()
    doc = _make_doc_mock(tables=[], texts=[])
    out = DoclingAdapter().egress(doc, ir, latency_ms=50)
    if out.payload.data is not None:
        assert "docling_table_count" not in out.payload.data


def test_egress_no_texts_entities_is_none() -> None:
    ir = _make_ir()
    doc = _make_doc_mock(texts=[])
    out = DoclingAdapter().egress(doc, ir, latency_ms=50)
    assert out.payload.entities is None


# ---------------------------------------------------------------------------
# egress — provenance
# ---------------------------------------------------------------------------

def test_egress_provenance_latency_ms_matches() -> None:
    ir = _make_ir()
    doc = _make_doc_mock(texts=[])
    out = DoclingAdapter().egress(doc, ir, latency_ms=333)
    assert out.provenance[-1].latency_ms == 333


def test_egress_provenance_model_id() -> None:
    ir = _make_ir()
    doc = _make_doc_mock(texts=[])
    out = DoclingAdapter().egress(doc, ir, latency_ms=50)
    assert out.provenance[-1].model_id == "docling-project/docling"


def test_egress_unknown_model_output_returns_safe_clone() -> None:
    ir = _make_ir()
    out = DoclingAdapter().egress(42, ir, latency_ms=50)
    assert isinstance(out, CanonicalIR)
    assert len(out.provenance) == len(ir.provenance) + 1


def test_egress_export_to_markdown_raises_falls_back() -> None:
    ir = _make_ir()
    doc = _make_doc_mock(markdown_raises=True, texts=[])
    out = DoclingAdapter().egress(doc, ir, latency_ms=50)
    assert out is not None
    # Falls back to export_to_text() result — must not raise.
    assert out.payload.content == "Fallback plain text."


def test_egress_original_ir_not_mutated() -> None:
    ir = _make_ir(content="/original/path.pdf")
    original_content = ir.payload.content
    original_entities = ir.payload.entities
    original_prov_len = len(ir.provenance)

    doc = _make_doc_mock(
        texts=[{"text": "Some text", "label": "paragraph"}],
        tables=[MagicMock()],
        markdown="# New content",
    )
    DoclingAdapter().egress(doc, ir, latency_ms=100)

    assert ir.payload.content == original_content
    assert ir.payload.entities == original_entities
    assert len(ir.provenance) == original_prov_len


def test_egress_provenance_confidence_is_1_0() -> None:
    ir = _make_ir()
    doc = _make_doc_mock(texts=[])
    out = DoclingAdapter().egress(doc, ir, latency_ms=50)
    assert out.provenance[-1].confidence == 1.0


# ---------------------------------------------------------------------------
# AdapterValidator — specific fixtures
# ---------------------------------------------------------------------------

def test_validator_legal_extract_basic() -> None:
    AdapterValidator(DoclingAdapter(), fixtures=[LEGAL_EXTRACT_BASIC]).assert_valid()


def test_validator_general_embed_large() -> None:
    AdapterValidator(DoclingAdapter(), fixtures=[GENERAL_EMBED_LARGE]).assert_valid()


# ---------------------------------------------------------------------------
# AdapterValidator — all 20 standard fixtures (§9 G-S06)
# ---------------------------------------------------------------------------

def test_validator_all_fixtures() -> None:
    """DoclingAdapter must pass all 13 MUST rules across all 20 standard fixtures."""
    AdapterValidator(DoclingAdapter(), fixtures=ALL_FIXTURES).assert_valid()


def test_validator_result_has_no_errors() -> None:
    result = AdapterValidator(DoclingAdapter(), fixtures=ALL_FIXTURES).run()
    assert result.passed, result.summary()
