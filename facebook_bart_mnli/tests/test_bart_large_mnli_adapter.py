"""Tests for BartLargeMnliAdapter — unit behaviour and full AdapterValidator suite."""

from __future__ import annotations

import time
import uuid
from typing import Any

import pytest

from synapse_sdk.testing import AdapterValidator
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
from bart_large_mnli_adapter import BartLargeMnliAdapter


# ---------------------------------------------------------------------------
# IR factory
# ---------------------------------------------------------------------------

def _make_ir(
    content: str | None = "one day I will see the world",
    candidate_labels: list[str] | None = None,
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
            domain=Domain.general,
            priority=2,
            latency_budget_ms=1000,
            candidate_labels=candidate_labels,
        ),
        payload=Payload(**payload_kwargs),
        compliance_envelope=compliance or ComplianceEnvelope(),
    )


# ---------------------------------------------------------------------------
# Standard mock output matching the verified HF schema
# ---------------------------------------------------------------------------

_MOCK_THREE = {
    "sequence": "one day I will see the world",
    "labels": ["travel", "dancing", "cooking"],
    "scores": [0.9938, 0.0032, 0.0029],
}


# ---------------------------------------------------------------------------
# GROUP A — Validator fixture suite (all standard fixtures, parametrized)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", ALL_FIXTURES)
def test_all_fixtures(fixture: CanonicalIR) -> None:
    AdapterValidator(BartLargeMnliAdapter()).assert_valid_on(fixture)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_model_id() -> None:
    assert BartLargeMnliAdapter().MODEL_ID == "facebook/bart-large-mnli"


def test_adapter_version_semver() -> None:
    ver = BartLargeMnliAdapter().ADAPTER_VERSION
    parts = ver.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_adapter_version_is_1_0_0() -> None:
    assert BartLargeMnliAdapter().ADAPTER_VERSION == "1.0.0"


# ---------------------------------------------------------------------------
# GROUP B — Egress correctness (mock output only)
# ---------------------------------------------------------------------------

def test_egress_returns_canonical_ir() -> None:
    out = BartLargeMnliAdapter().egress(_MOCK_THREE, _make_ir(), latency_ms=100)
    assert isinstance(out, CanonicalIR)


def test_egress_all_three_labels_stored() -> None:
    out = BartLargeMnliAdapter().egress(_MOCK_THREE, _make_ir(), latency_ms=100)
    assert out.payload.labels is not None
    assert len(out.payload.labels) == 3


def test_egress_labels_length_equals_candidate_count() -> None:
    out = BartLargeMnliAdapter().egress(_MOCK_THREE, _make_ir(), latency_ms=100)
    assert len(out.payload.labels) == len(_MOCK_THREE["labels"])  # type: ignore[arg-type]


def test_egress_top_label_is_travel() -> None:
    out = BartLargeMnliAdapter().egress(_MOCK_THREE, _make_ir(), latency_ms=100)
    assert out.payload.labels[0].label == "travel"  # type: ignore[index]


def test_egress_top_score_is_0_9938() -> None:
    out = BartLargeMnliAdapter().egress(_MOCK_THREE, _make_ir(), latency_ms=100)
    assert out.payload.labels[0].score == pytest.approx(0.9938, abs=1e-6)  # type: ignore[index]


def test_egress_second_label_is_dancing() -> None:
    out = BartLargeMnliAdapter().egress(_MOCK_THREE, _make_ir(), latency_ms=100)
    assert out.payload.labels[1].label == "dancing"  # type: ignore[index]


def test_egress_third_label_is_cooking() -> None:
    out = BartLargeMnliAdapter().egress(_MOCK_THREE, _make_ir(), latency_ms=100)
    assert out.payload.labels[2].label == "cooking"  # type: ignore[index]


def test_egress_confidence_equals_top_score() -> None:
    out = BartLargeMnliAdapter().egress(_MOCK_THREE, _make_ir(), latency_ms=100)
    assert out.provenance[-1].confidence == pytest.approx(0.9938, abs=1e-6)


def test_egress_provenance_length_is_original_plus_one() -> None:
    ir = _make_ir()
    out = BartLargeMnliAdapter().egress(_MOCK_THREE, ir, latency_ms=100)
    assert len(out.provenance) == len(ir.provenance) + 1


def test_egress_task_header_carried_forward() -> None:
    ir = _make_ir(candidate_labels=["travel", "cooking", "dancing"])
    out = BartLargeMnliAdapter().egress(_MOCK_THREE, ir, latency_ms=100)
    assert out.task_header.model_dump() == ir.task_header.model_dump()


def test_egress_compliance_envelope_carried_forward() -> None:
    compliance = ComplianceEnvelope(
        required_tags=["gdpr"],
        pii_present=False,
        retention_policy="30d",
    )
    ir = _make_ir(compliance=compliance)
    out = BartLargeMnliAdapter().egress(_MOCK_THREE, ir, latency_ms=100)
    assert out.compliance_envelope.model_dump() == compliance.model_dump()


def test_egress_pii_present_remains_false() -> None:
    ir = _make_ir(compliance=ComplianceEnvelope(pii_present=False))
    out = BartLargeMnliAdapter().egress(_MOCK_THREE, ir, latency_ms=100)
    assert out.compliance_envelope.pii_present is False


def test_egress_pii_present_never_set_true() -> None:
    ir = _make_ir()
    out = BartLargeMnliAdapter().egress(_MOCK_THREE, ir, latency_ms=100)
    assert out.compliance_envelope.pii_present is not True


def test_egress_latency_ms_stored_correctly() -> None:
    out = BartLargeMnliAdapter().egress(_MOCK_THREE, _make_ir(), latency_ms=350)
    assert out.provenance[-1].latency_ms == 350


def test_egress_two_label_output_stored_correctly() -> None:
    mock = {
        "sequence": "test",
        "labels": ["travel", "cooking"],
        "scores": [0.8, 0.2],
    }
    out = BartLargeMnliAdapter().egress(mock, _make_ir(), latency_ms=100)
    assert len(out.payload.labels) == 2  # type: ignore[arg-type]
    assert out.payload.labels[0].label == "travel"  # type: ignore[index]
    assert out.payload.labels[1].label == "cooking"  # type: ignore[index]


def test_egress_single_label_output_stored_correctly() -> None:
    mock = {
        "sequence": "test",
        "labels": ["travel"],
        "scores": [1.0],
    }
    out = BartLargeMnliAdapter().egress(mock, _make_ir(), latency_ms=100)
    assert len(out.payload.labels) == 1  # type: ignore[arg-type]
    assert out.payload.labels[0].label == "travel"  # type: ignore[index]
    assert out.payload.labels[0].score == pytest.approx(1.0)  # type: ignore[index]


def test_egress_payload_labels_items_are_classification_instances() -> None:
    out = BartLargeMnliAdapter().egress(_MOCK_THREE, _make_ir(), latency_ms=100)
    for item in out.payload.labels:  # type: ignore[union-attr]
        assert isinstance(item, Classification)


def test_egress_does_not_mutate_original_ir() -> None:
    ir = _make_ir(content="one day I will see the world")
    original_content = ir.payload.content
    BartLargeMnliAdapter().egress(_MOCK_THREE, ir, latency_ms=100)
    assert ir.payload.content == original_content
    assert ir.payload.labels is None


def test_egress_existing_provenance_not_modified() -> None:
    prior = ProvenanceEntry(
        model_id="some/prior-model",
        adapter_version="1.0.0",
        confidence=0.72,
        latency_ms=300,
        timestamp_unix=int(time.time()),
    )
    ir = _make_ir()
    ir.provenance.append(prior)

    out = BartLargeMnliAdapter().egress(_MOCK_THREE, ir, latency_ms=100)

    assert out.provenance[0].model_id == "some/prior-model"
    assert out.provenance[0].confidence == 0.72
    assert len(out.provenance) == 2
    assert out.provenance[1].model_id == "facebook/bart-large-mnli"


def test_egress_provenance_model_id_matches() -> None:
    out = BartLargeMnliAdapter().egress(_MOCK_THREE, _make_ir(), latency_ms=100)
    assert out.provenance[-1].model_id == "facebook/bart-large-mnli"


# ---------------------------------------------------------------------------
# GROUP C — Ingress correctness
# ---------------------------------------------------------------------------

def test_ingress_returns_dict() -> None:
    result = BartLargeMnliAdapter().ingress(_make_ir())
    assert isinstance(result, dict)


def test_ingress_candidate_labels_present_passed_through() -> None:
    labels = ["travel", "cooking", "dancing"]
    ir = _make_ir(candidate_labels=labels)
    result = BartLargeMnliAdapter().ingress(ir)
    assert result["candidate_labels"] == labels


def test_ingress_candidate_labels_none_uses_default() -> None:
    ir = _make_ir(candidate_labels=None)
    result = BartLargeMnliAdapter().ingress(ir)
    assert result["candidate_labels"] == ["positive", "negative", "neutral"]


def test_ingress_candidate_labels_empty_list_uses_default() -> None:
    ir = _make_ir(candidate_labels=[])
    result = BartLargeMnliAdapter().ingress(ir)
    assert result["candidate_labels"] == ["positive", "negative", "neutral"]


def test_ingress_content_stored_under_text() -> None:
    ir = _make_ir(content="Hello world!")
    result = BartLargeMnliAdapter().ingress(ir)
    assert result["text"] == "Hello world!"


def test_ingress_content_none_returns_empty_text() -> None:
    ir = _make_ir(content=None, modality="structured")
    result = BartLargeMnliAdapter().ingress(ir)
    assert result["text"] == ""


def test_ingress_output_has_text_and_candidate_labels_keys() -> None:
    result = BartLargeMnliAdapter().ingress(_make_ir())
    assert "text" in result
    assert "candidate_labels" in result


def test_ingress_never_returns_none() -> None:
    adapter = BartLargeMnliAdapter()
    assert adapter.ingress(_make_ir()) is not None
    assert adapter.ingress(_make_ir(content=None, modality="structured")) is not None


# ---------------------------------------------------------------------------
# GROUP D — Edge cases (defensive egress)
# ---------------------------------------------------------------------------

def test_egress_none_output_no_crash() -> None:
    out = BartLargeMnliAdapter().egress(None, _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)


def test_egress_none_output_empty_labels() -> None:
    out = BartLargeMnliAdapter().egress(None, _make_ir(), latency_ms=0)
    assert out.payload.labels == []


def test_egress_none_output_confidence_zero() -> None:
    out = BartLargeMnliAdapter().egress(None, _make_ir(), latency_ms=0)
    assert out.provenance[-1].confidence == 0.0


def test_egress_empty_dict_no_crash() -> None:
    out = BartLargeMnliAdapter().egress({}, _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)


def test_egress_empty_dict_empty_labels() -> None:
    out = BartLargeMnliAdapter().egress({}, _make_ir(), latency_ms=0)
    assert out.payload.labels == []


def test_egress_missing_labels_key_no_crash() -> None:
    out = BartLargeMnliAdapter().egress({"scores": [0.9, 0.1]}, _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)


def test_egress_missing_scores_key_no_crash() -> None:
    out = BartLargeMnliAdapter().egress({"labels": ["travel", "cooking"]}, _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)


def test_egress_missing_scores_key_empty_labels() -> None:
    out = BartLargeMnliAdapter().egress({"labels": ["travel", "cooking"]}, _make_ir(), latency_ms=0)
    assert out.payload.labels == []


def test_egress_mismatched_lengths_no_crash() -> None:
    mock = {"labels": ["travel", "cooking", "dancing"], "scores": [0.9]}
    out = BartLargeMnliAdapter().egress(mock, _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)


def test_egress_mismatched_lengths_zip_truncates() -> None:
    mock = {"labels": ["travel", "cooking", "dancing"], "scores": [0.9]}
    out = BartLargeMnliAdapter().egress(mock, _make_ir(), latency_ms=0)
    assert len(out.payload.labels) == 1  # type: ignore[arg-type]


def test_egress_empty_labels_list_empty_payload_labels() -> None:
    mock = {"sequence": "test", "labels": [], "scores": []}
    out = BartLargeMnliAdapter().egress(mock, _make_ir(), latency_ms=0)
    assert out.payload.labels == []


def test_egress_empty_labels_list_confidence_zero() -> None:
    mock = {"sequence": "test", "labels": [], "scores": []}
    out = BartLargeMnliAdapter().egress(mock, _make_ir(), latency_ms=0)
    assert out.provenance[-1].confidence == 0.0


def test_egress_latency_ms_zero_valid_provenance() -> None:
    out = BartLargeMnliAdapter().egress(_MOCK_THREE, _make_ir(), latency_ms=0)
    assert out.provenance[-1].latency_ms == 0
    assert 0.0 <= out.provenance[-1].confidence <= 1.0


def test_egress_provenance_always_appended_even_on_bad_output() -> None:
    ir = _make_ir()
    out = BartLargeMnliAdapter().egress(None, ir, latency_ms=0)
    assert len(out.provenance) == len(ir.provenance) + 1


def test_egress_list_output_no_crash() -> None:
    out = BartLargeMnliAdapter().egress([{"label": "travel", "score": 0.9}], _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)


def test_egress_list_output_empty_labels() -> None:
    out = BartLargeMnliAdapter().egress([{"label": "travel", "score": 0.9}], _make_ir(), latency_ms=0)
    assert out.payload.labels == []


def test_egress_empty_compliance_envelope_carried_forward() -> None:
    ir = _make_ir(compliance=ComplianceEnvelope())
    out = BartLargeMnliAdapter().egress(_MOCK_THREE, ir, latency_ms=100)
    assert out.compliance_envelope.model_dump() == ComplianceEnvelope().model_dump()


# ---------------------------------------------------------------------------
# GROUP E — AdapterValidator direct pass (non-parametrized)
# ---------------------------------------------------------------------------

def test_validator_passes() -> None:
    AdapterValidator(BartLargeMnliAdapter()).assert_valid()
