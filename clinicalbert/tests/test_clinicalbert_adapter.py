"""Tests for ClinicalBertAdapter unit behavior and validator coverage."""

from __future__ import annotations

import time
import uuid
from typing import Any

import pytest

from clinicalbert_adapter import ClinicalBertAdapter
from synapse_sdk.testing.fixtures import ALL_FIXTURES
from synapse_sdk.types import (
    CanonicalIR,
    Classification,
    ComplianceEnvelope,
    Domain,
    Payload,
    ProvenanceEntry,
    TaskHeader,
    TaskType,
)
from synapse_sdk.validator import AdapterValidator


def _make_ir(
    content: str | None = "The patient reports chest [MASK] after exercise.",
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
            task_type=TaskType.classify,
            domain=Domain.medical,
            priority=2,
            latency_budget_ms=1000,
        ),
        payload=Payload(**payload_kwargs),
        compliance_envelope=compliance or ComplianceEnvelope(),
    )


def _mock_output() -> list[dict[str, Any]]:
    return [
        {
            "score": 0.8497790098190308,
            "sequence": "the patient reports chest pain after exercise.",
            "token": 38576,
            "token_str": "pain",
        },
        {
            "score": 0.09418246150016785,
            "sequence": "the patient reports chest pressure after exercise.",
            "token": 23460,
            "token_str": "pressure",
        },
    ]


@pytest.mark.parametrize("fixture", ALL_FIXTURES)
def test_all_fixtures(fixture: CanonicalIR) -> None:
    AdapterValidator(ClinicalBertAdapter()).assert_valid_on(fixture)


def test_model_id() -> None:
    assert ClinicalBertAdapter().MODEL_ID == "medicalai/ClinicalBERT"


def test_adapter_version_semver() -> None:
    ver = ClinicalBertAdapter().ADAPTER_VERSION
    parts = ver.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
    assert ver == "1.0.0"


def test_ingress_returns_text_only() -> None:
    result = ClinicalBertAdapter().ingress(_make_ir())
    assert result == {"text": "The patient reports chest [MASK] after exercise."}


def test_ingress_preserves_mask_token_exactly() -> None:
    text = "Laboratory results showed elevated [MASK] levels."
    assert ClinicalBertAdapter().ingress(_make_ir(content=text))["text"] == text


def test_ingress_empty_content_returns_empty_text() -> None:
    assert ClinicalBertAdapter().ingress(_make_ir(content="")) == {"text": ""}


def test_ingress_none_content_returns_empty_text() -> None:
    assert ClinicalBertAdapter().ingress(_make_ir(content=None, modality="structured")) == {
        "text": ""
    }


def test_ingress_structured_payload_returns_empty_text() -> None:
    assert ClinicalBertAdapter().ingress(_make_ir(modality="structured")) == {"text": ""}


def test_egress_returns_canonical_ir() -> None:
    out = ClinicalBertAdapter().egress(_mock_output(), _make_ir(), latency_ms=20)
    assert isinstance(out, CanonicalIR)


def test_egress_converts_candidates_to_classifications() -> None:
    out = ClinicalBertAdapter().egress(_mock_output(), _make_ir(), latency_ms=20)
    assert out.payload.labels is not None
    assert len(out.payload.labels) == 2
    assert all(isinstance(label, Classification) for label in out.payload.labels)
    assert [label.label for label in out.payload.labels] == ["pain", "pressure"]
    assert out.payload.labels[0].score == pytest.approx(0.8497790098190308)
    assert out.payload.labels[1].score == pytest.approx(0.09418246150016785)


def test_egress_preserves_payload_content() -> None:
    ir = _make_ir(content="The patient was prescribed [MASK] for hypertension.")
    out = ClinicalBertAdapter().egress(_mock_output(), ir, latency_ms=20)
    assert out.payload.content == ir.payload.content


def test_egress_confidence_uses_first_valid_candidate_score() -> None:
    out = ClinicalBertAdapter().egress(_mock_output(), _make_ir(), latency_ms=20)
    assert out.provenance[-1].confidence == pytest.approx(0.8497790098190308)


def test_egress_appends_exactly_one_provenance_entry() -> None:
    ir = _make_ir()
    out = ClinicalBertAdapter().egress(_mock_output(), ir, latency_ms=20)
    assert len(out.provenance) == len(ir.provenance) + 1
    assert out.provenance[-1].model_id == "medicalai/ClinicalBERT"


def test_egress_preserves_existing_provenance_entries() -> None:
    prior = ProvenanceEntry(
        model_id="upstream/model",
        adapter_version="1.2.3",
        confidence=0.77,
        latency_ms=42,
        timestamp_unix=int(time.time()),
    )
    ir = _make_ir()
    ir.provenance.append(prior)
    out = ClinicalBertAdapter().egress(_mock_output(), ir, latency_ms=20)
    assert out.provenance[0].model_id == "upstream/model"
    assert out.provenance[0].confidence == 0.77
    assert len(out.provenance) == 2


def test_egress_preserves_task_header_and_compliance_envelope() -> None:
    compliance = ComplianceEnvelope(
        required_tags=["hipaa"],
        pii_present=False,
        retention_policy="30d",
        data_residency=["us"],
        purpose_limitation="clinical-decision-support",
    )
    ir = _make_ir(compliance=compliance)
    out = ClinicalBertAdapter().egress(_mock_output(), ir, latency_ms=20)
    assert out.task_header.model_dump() == ir.task_header.model_dump()
    assert out.compliance_envelope.model_dump() == compliance.model_dump()
    assert out.compliance_envelope.pii_present is False


@pytest.mark.parametrize("model_output", [None, {}, [], "pain"])
def test_egress_malformed_top_level_output_returns_empty_labels(model_output: Any) -> None:
    out = ClinicalBertAdapter().egress(model_output, _make_ir(), latency_ms=0)
    assert out.payload.labels == []
    assert out.provenance[-1].confidence == 0.0


def test_egress_skips_malformed_candidates_and_keeps_valid_candidates() -> None:
    model_output = [
        "pain",
        {"token_str": "missing_score"},
        {"score": "not-numeric", "token_str": "bad_score"},
        {"score": 0.25, "token_str": "valid"},
    ]
    out = ClinicalBertAdapter().egress(model_output, _make_ir(), latency_ms=0)
    assert out.payload.labels is not None
    assert [label.label for label in out.payload.labels] == ["valid"]
    assert out.payload.labels[0].score == pytest.approx(0.25)
    assert out.provenance[-1].confidence == pytest.approx(0.25)


def test_egress_missing_token_str_is_skipped() -> None:
    out = ClinicalBertAdapter().egress([{"score": 0.5}], _make_ir(), latency_ms=0)
    assert out.payload.labels == []


def test_egress_does_not_mutate_original_ir() -> None:
    ir = _make_ir()
    ClinicalBertAdapter().egress(_mock_output(), ir, latency_ms=20)
    assert ir.payload.labels is None
    assert len(ir.provenance) == 0


def test_validator_passes() -> None:
    AdapterValidator(ClinicalBertAdapter()).assert_valid()
