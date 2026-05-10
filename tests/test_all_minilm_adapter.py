"""Tests for AllMiniLML6V2Adapter — unit behaviour and full AdapterValidator suite."""

from __future__ import annotations

import time
import uuid
from typing import Any

import numpy as np
import pytest

from all_minilm_adapter import AllMiniLML6V2Adapter
from synapse_sdk.testing.fixtures import (
    ALL_FIXTURES,
    GENERAL_EMBED_LARGE,
    LEGAL_EXTRACT_BASIC,
)
from synapse_sdk.types import (
    CanonicalIR,
    ComplianceEnvelope,
    Domain,
    Payload,
    ProvenanceEntry,
    TaskHeader,
    TaskType,
)
from synapse_sdk.validator import AdapterValidator


# ---------------------------------------------------------------------------
# IR factory
# ---------------------------------------------------------------------------

def _make_ir(
    content: str | None = "The quick brown fox jumps over the lazy dog.",
    modality: str = "text",
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
            task_type=TaskType.embed,
            domain=Domain.general,
            priority=2,
            latency_budget_ms=500,
        ),
        payload=Payload(**payload_kwargs),
        compliance_envelope=compliance or ComplianceEnvelope(),
    )


# ---------------------------------------------------------------------------
# Mock numpy output factory
# ---------------------------------------------------------------------------

def _make_output(rows: int = 1, cols: int = 384) -> np.ndarray:
    return np.random.rand(rows, cols).astype(np.float32)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_model_id() -> None:
    assert AllMiniLML6V2Adapter().MODEL_ID == "sentence-transformers/all-MiniLM-L6-v2"


def test_adapter_version_semver() -> None:
    ver = AllMiniLML6V2Adapter().ADAPTER_VERSION
    parts = ver.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_embedding_dim_constant() -> None:
    assert AllMiniLML6V2Adapter.EMBEDDING_DIM == 384


# ---------------------------------------------------------------------------
# ingress — standard cases
# ---------------------------------------------------------------------------

def test_ingress_standard_text() -> None:
    ir = _make_ir(content="Hello world")
    result = AllMiniLML6V2Adapter().ingress(ir)
    assert result == {"sentences": ["Hello world"]}


def test_ingress_none_content_returns_empty_string() -> None:
    ir = _make_ir(content=None, modality="structured")
    result = AllMiniLML6V2Adapter().ingress(ir)
    assert result == {"sentences": [""]}


def test_ingress_empty_string_content() -> None:
    ir = _make_ir(content="")
    result = AllMiniLML6V2Adapter().ingress(ir)
    assert result == {"sentences": [""]}


def test_ingress_never_returns_none() -> None:
    adapter = AllMiniLML6V2Adapter()
    assert adapter.ingress(_make_ir()) is not None
    assert adapter.ingress(_make_ir(content=None, modality="structured")) is not None


def test_ingress_returns_dict() -> None:
    result = AllMiniLML6V2Adapter().ingress(_make_ir())
    assert isinstance(result, dict)


def test_ingress_output_has_sentences_key() -> None:
    result = AllMiniLML6V2Adapter().ingress(_make_ir())
    assert "sentences" in result
    assert isinstance(result["sentences"], list)


def test_ingress_sentences_list_has_exactly_one_item() -> None:
    result = AllMiniLML6V2Adapter().ingress(_make_ir())
    assert len(result["sentences"]) == 1


def test_ingress_multiline_text_preserved() -> None:
    text = "Line one.\nLine two.\nLine three."
    ir = _make_ir(content=text)
    result = AllMiniLML6V2Adapter().ingress(ir)
    assert result["sentences"][0] == text


# ---------------------------------------------------------------------------
# egress — return type and payload
# ---------------------------------------------------------------------------

def test_egress_returns_canonical_ir() -> None:
    out = AllMiniLML6V2Adapter().egress(_make_output(), _make_ir(), latency_ms=10)
    assert isinstance(out, CanonicalIR)


def test_egress_sets_modality_to_embedding() -> None:
    out = AllMiniLML6V2Adapter().egress(_make_output(), _make_ir(), latency_ms=10)
    assert out.payload.modality == "embedding"


def test_egress_vector_is_list_of_384_floats() -> None:
    out = AllMiniLML6V2Adapter().egress(_make_output(), _make_ir(), latency_ms=10)
    assert isinstance(out.payload.vector, list)
    assert len(out.payload.vector) == 384


def test_egress_vector_dim_is_384() -> None:
    out = AllMiniLML6V2Adapter().egress(_make_output(), _make_ir(), latency_ms=10)
    assert out.payload.vector_dim == 384


def test_egress_embedding_model_field() -> None:
    out = AllMiniLML6V2Adapter().egress(_make_output(), _make_ir(), latency_ms=10)
    assert out.payload.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"


def test_egress_vector_values_are_python_floats_not_numpy() -> None:
    out = AllMiniLML6V2Adapter().egress(_make_output(), _make_ir(), latency_ms=10)
    assert out.payload.vector is not None
    for val in out.payload.vector:
        assert isinstance(val, float), f"Expected float, got {type(val)}"
        assert not isinstance(val, np.floating), "Should be Python float, not numpy float"


def test_egress_vector_values_match_input_array() -> None:
    arr = _make_output()
    out = AllMiniLML6V2Adapter().egress(arr, _make_ir(), latency_ms=10)
    assert out.payload.vector is not None
    assert pytest.approx(out.payload.vector) == arr[0].tolist()


# ---------------------------------------------------------------------------
# egress — provenance
# ---------------------------------------------------------------------------

def test_egress_appends_exactly_one_provenance_entry() -> None:
    ir = _make_ir()
    out = AllMiniLML6V2Adapter().egress(_make_output(), ir, latency_ms=10)
    assert len(out.provenance) == len(ir.provenance) + 1


def test_egress_provenance_confidence_is_1_0() -> None:
    out = AllMiniLML6V2Adapter().egress(_make_output(), _make_ir(), latency_ms=10)
    assert out.provenance[-1].confidence == 1.0


def test_egress_provenance_model_id_matches() -> None:
    out = AllMiniLML6V2Adapter().egress(_make_output(), _make_ir(), latency_ms=10)
    assert out.provenance[-1].model_id == "sentence-transformers/all-MiniLM-L6-v2"


def test_egress_existing_provenance_not_modified() -> None:
    prior = ProvenanceEntry(
        model_id="some/prior-model",
        adapter_version="2.3.1",
        confidence=0.75,
        latency_ms=300,
        timestamp_unix=int(time.time()),
    )
    ir = _make_ir()
    ir.provenance.append(prior)

    out = AllMiniLML6V2Adapter().egress(_make_output(), ir, latency_ms=10)

    assert out.provenance[0].model_id == "some/prior-model"
    assert out.provenance[0].confidence == 0.75
    assert len(out.provenance) == 2
    assert out.provenance[1].model_id == "sentence-transformers/all-MiniLM-L6-v2"


def test_egress_provenance_immutable_object_unchanged() -> None:
    prior = ProvenanceEntry(
        model_id="another/model",
        adapter_version="1.0.0",
        confidence=0.9,
        latency_ms=100,
        timestamp_unix=int(time.time()),
    )
    ir = _make_ir()
    ir.provenance.append(prior)
    snapshot_model_id = prior.model_id
    snapshot_confidence = prior.confidence

    AllMiniLML6V2Adapter().egress(_make_output(), ir, latency_ms=10)

    assert prior.model_id == snapshot_model_id
    assert prior.confidence == snapshot_confidence


# ---------------------------------------------------------------------------
# egress — task_header and compliance carried forward
# ---------------------------------------------------------------------------

def test_egress_task_header_carried_forward_unchanged() -> None:
    ir = _make_ir()
    out = AllMiniLML6V2Adapter().egress(_make_output(), ir, latency_ms=10)
    assert out.task_header.model_dump() == ir.task_header.model_dump()


def test_egress_compliance_envelope_carried_forward_unchanged() -> None:
    compliance = ComplianceEnvelope(
        required_tags=["gdpr"],
        pii_present=False,
        retention_policy="90d",
        data_residency=["eu-west-1"],
        purpose_limitation="analytics",
    )
    ir = _make_ir(compliance=compliance)
    out = AllMiniLML6V2Adapter().egress(_make_output(), ir, latency_ms=10)
    assert out.compliance_envelope.model_dump() == compliance.model_dump()


def test_egress_compliance_envelope_with_all_none_fields_carried() -> None:
    ir = _make_ir(compliance=ComplianceEnvelope())
    out = AllMiniLML6V2Adapter().egress(_make_output(), ir, latency_ms=10)
    assert out.compliance_envelope.model_dump() == ComplianceEnvelope().model_dump()


# ---------------------------------------------------------------------------
# egress — edge cases (graceful degradation, no raise)
# ---------------------------------------------------------------------------

def test_egress_empty_array_returns_empty_vector() -> None:
    empty = np.array([]).reshape(0, 384).astype(np.float32)
    out = AllMiniLML6V2Adapter().egress(empty, _make_ir(), latency_ms=10)
    assert isinstance(out, CanonicalIR)
    assert out.payload.vector == []
    assert out.payload.vector_dim == 0


def test_egress_wrong_dim_array_returns_empty_vector() -> None:
    wrong = np.random.rand(1, 128).astype(np.float32)
    out = AllMiniLML6V2Adapter().egress(wrong, _make_ir(), latency_ms=10)
    assert out.payload.vector == []
    assert out.payload.vector_dim == 0


def test_egress_none_output_does_not_raise() -> None:
    out = AllMiniLML6V2Adapter().egress(None, _make_ir(), latency_ms=10)
    assert isinstance(out, CanonicalIR)
    assert out.payload.vector == []
    assert out.payload.vector_dim == 0


def test_egress_dict_output_does_not_raise() -> None:
    out = AllMiniLML6V2Adapter().egress({"bad": "output"}, _make_ir(), latency_ms=10)
    assert isinstance(out, CanonicalIR)
    assert out.payload.vector == []


def test_egress_1d_array_does_not_raise() -> None:
    bad = np.random.rand(384).astype(np.float32)
    out = AllMiniLML6V2Adapter().egress(bad, _make_ir(), latency_ms=10)
    assert isinstance(out, CanonicalIR)
    assert out.payload.vector == []


def test_egress_empty_output_still_appends_provenance() -> None:
    empty = np.array([]).reshape(0, 384).astype(np.float32)
    ir = _make_ir()
    out = AllMiniLML6V2Adapter().egress(empty, ir, latency_ms=10)
    assert len(out.provenance) == len(ir.provenance) + 1


def test_egress_modality_always_set_to_embedding() -> None:
    ir = _make_ir(modality="structured")
    out = AllMiniLML6V2Adapter().egress(_make_output(), ir, latency_ms=10)
    assert out.payload.modality == "embedding"


def test_egress_does_not_overwrite_original_ir_payload() -> None:
    ir = _make_ir(content="Original text.")
    AllMiniLML6V2Adapter().egress(_make_output(), ir, latency_ms=10)
    assert ir.payload.modality == "text"
    assert ir.payload.content == "Original text."


# ---------------------------------------------------------------------------
# AdapterValidator — specific fixtures
# ---------------------------------------------------------------------------

def test_validator_legal_extract_basic() -> None:
    AdapterValidator(AllMiniLML6V2Adapter(), fixtures=[LEGAL_EXTRACT_BASIC]).assert_valid()


def test_validator_general_embed_large() -> None:
    AdapterValidator(AllMiniLML6V2Adapter(), fixtures=[GENERAL_EMBED_LARGE]).assert_valid()


# ---------------------------------------------------------------------------
# AdapterValidator — all 20 standard fixtures
# ---------------------------------------------------------------------------

def test_validator_all_fixtures() -> None:
    """AllMiniLML6V2Adapter must pass all 13 MUST rules across all 20 standard fixtures."""
    AdapterValidator(AllMiniLML6V2Adapter(), fixtures=ALL_FIXTURES).assert_valid()


def test_validator_result_has_no_errors() -> None:
    result = AdapterValidator(AllMiniLML6V2Adapter(), fixtures=ALL_FIXTURES).run()
    assert result.passed, result.summary()
