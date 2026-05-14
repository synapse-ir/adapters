"""Tests for FinbertAdapter — unit behaviour and full AdapterValidator suite."""

from __future__ import annotations

import time
import uuid
from typing import Any

import pytest

from finbert_adapter import FinbertAdapter
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


# ---------------------------------------------------------------------------
# IR factory
# ---------------------------------------------------------------------------

def _make_ir(
    content: str | None = "Revenues increased significantly this quarter.",
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
            domain=Domain.finance,
            priority=2,
            latency_budget_ms=1000,
        ),
        payload=Payload(**payload_kwargs),
        compliance_envelope=compliance or ComplianceEnvelope(),
    )


# ---------------------------------------------------------------------------
# Mock pipeline output factory
# ---------------------------------------------------------------------------

def _mock_output(
    label: str = "positive",
    score: float = 0.9123,
) -> list[dict[str, Any]]:
    return [{"label": label, "score": score}]


# ---------------------------------------------------------------------------
# GROUP A — Validator fixture suite (all 20 canonical fixtures)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", ALL_FIXTURES)
def test_all_fixtures(fixture: CanonicalIR) -> None:
    AdapterValidator(FinbertAdapter()).assert_valid_on(fixture)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_model_id() -> None:
    assert FinbertAdapter().MODEL_ID == "ProsusAI/finbert"


def test_adapter_version_semver() -> None:
    ver = FinbertAdapter().ADAPTER_VERSION
    parts = ver.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_adapter_version_is_1_0_0() -> None:
    assert FinbertAdapter().ADAPTER_VERSION == "1.0.0"


# ---------------------------------------------------------------------------
# GROUP C — Ingress correctness
# ---------------------------------------------------------------------------

def test_ingress_returns_dict() -> None:
    result = FinbertAdapter().ingress(_make_ir())
    assert isinstance(result, dict)


def test_ingress_non_empty_content_returned_as_text() -> None:
    ir = _make_ir(content="Net income rose 12% year-over-year.")
    result = FinbertAdapter().ingress(ir)
    assert result["text"] == "Net income rose 12% year-over-year."


def test_ingress_empty_string_content_returns_empty_text() -> None:
    ir = _make_ir(content="")
    result = FinbertAdapter().ingress(ir)
    assert result["text"] == ""


def test_ingress_none_content_returns_empty_text() -> None:
    ir = _make_ir(content=None, modality="structured")
    result = FinbertAdapter().ingress(ir)
    assert result["text"] == ""


def test_ingress_has_only_text_key() -> None:
    result = FinbertAdapter().ingress(_make_ir())
    assert set(result.keys()) == {"text"}


def test_ingress_never_returns_none() -> None:
    assert FinbertAdapter().ingress(_make_ir()) is not None
    assert FinbertAdapter().ingress(_make_ir(content=None, modality="structured")) is not None


# ---------------------------------------------------------------------------
# GROUP B — Egress correctness with mock output
# ---------------------------------------------------------------------------

def test_egress_returns_canonical_ir() -> None:
    out = FinbertAdapter().egress(_mock_output(), _make_ir(), latency_ms=100)
    assert isinstance(out, CanonicalIR)


def test_egress_positive_label_stored_correctly() -> None:
    out = FinbertAdapter().egress(_mock_output("positive", 0.9723), _make_ir(), latency_ms=100)
    assert out.payload.labels is not None
    assert out.payload.labels[0].label == "positive"


def test_egress_positive_score_stored_correctly() -> None:
    out = FinbertAdapter().egress(_mock_output("positive", 0.9723), _make_ir(), latency_ms=100)
    assert out.payload.labels is not None
    assert out.payload.labels[0].score == pytest.approx(0.9723)


def test_egress_negative_label_stored_correctly() -> None:
    out = FinbertAdapter().egress(_mock_output("negative", 0.8811), _make_ir(), latency_ms=100)
    assert out.payload.labels is not None
    assert out.payload.labels[0].label == "negative"


def test_egress_neutral_label_stored_correctly() -> None:
    out = FinbertAdapter().egress(_mock_output("neutral", 0.7500), _make_ir(), latency_ms=100)
    assert out.payload.labels is not None
    assert out.payload.labels[0].label == "neutral"


def test_egress_high_confidence_score_stored_correctly() -> None:
    out = FinbertAdapter().egress(_mock_output("positive", 0.99), _make_ir(), latency_ms=100)
    assert out.payload.labels is not None
    assert out.payload.labels[0].score == pytest.approx(0.99)


def test_egress_low_confidence_score_stored_correctly() -> None:
    out = FinbertAdapter().egress(_mock_output("neutral", 0.01), _make_ir(), latency_ms=100)
    assert out.payload.labels is not None
    assert out.payload.labels[0].score == pytest.approx(0.01)


def test_egress_confidence_in_provenance_matches_top_prediction_score() -> None:
    score = 0.8542
    out = FinbertAdapter().egress(_mock_output("positive", score), _make_ir(), latency_ms=100)
    assert out.provenance[-1].confidence == pytest.approx(score)


def test_egress_provenance_list_length_is_original_plus_one() -> None:
    ir = _make_ir()
    original_len = len(ir.provenance)
    out = FinbertAdapter().egress(_mock_output(), ir, latency_ms=100)
    assert len(out.provenance) == original_len + 1


def test_egress_task_header_carried_forward_unchanged() -> None:
    ir = _make_ir()
    out = FinbertAdapter().egress(_mock_output(), ir, latency_ms=100)
    assert out.task_header.model_dump() == ir.task_header.model_dump()


def test_egress_compliance_envelope_carried_forward_unchanged() -> None:
    compliance = ComplianceEnvelope(
        required_tags=["sox", "pci"],
        pii_present=False,
        retention_policy="3y",
        data_residency=["us-east-1"],
        purpose_limitation="financial-reporting",
    )
    ir = _make_ir(compliance=compliance)
    out = FinbertAdapter().egress(_mock_output(), ir, latency_ms=100)
    assert out.compliance_envelope.model_dump() == compliance.model_dump()


def test_egress_pii_present_remains_false() -> None:
    compliance = ComplianceEnvelope(pii_present=False)
    ir = _make_ir(compliance=compliance)
    out = FinbertAdapter().egress(_mock_output(), ir, latency_ms=100)
    assert out.compliance_envelope.pii_present is False


def test_egress_pii_present_never_set_true_by_adapter() -> None:
    ir = _make_ir(compliance=ComplianceEnvelope())
    out = FinbertAdapter().egress(_mock_output("positive", 0.99), ir, latency_ms=100)
    assert out.compliance_envelope.pii_present is not True


def test_egress_payload_labels_is_list_of_length_one() -> None:
    out = FinbertAdapter().egress(_mock_output(), _make_ir(), latency_ms=100)
    assert isinstance(out.payload.labels, list)
    assert len(out.payload.labels) == 1


def test_egress_payload_labels_item_is_classification() -> None:
    out = FinbertAdapter().egress(_mock_output(), _make_ir(), latency_ms=100)
    assert out.payload.labels is not None
    assert isinstance(out.payload.labels[0], Classification)


def test_egress_payload_labels_first_label_matches_mock_output() -> None:
    out = FinbertAdapter().egress(_mock_output("negative", 0.75), _make_ir(), latency_ms=100)
    assert out.payload.labels is not None
    assert out.payload.labels[0].label == "negative"


def test_egress_payload_labels_first_score_matches_mock_output() -> None:
    out = FinbertAdapter().egress(_mock_output("neutral", 0.60), _make_ir(), latency_ms=100)
    assert out.payload.labels is not None
    assert out.payload.labels[0].score == pytest.approx(0.60)


def test_egress_provenance_model_id_matches() -> None:
    out = FinbertAdapter().egress(_mock_output(), _make_ir(), latency_ms=100)
    assert out.provenance[-1].model_id == "ProsusAI/finbert"


def test_egress_existing_provenance_not_modified() -> None:
    prior = ProvenanceEntry(
        model_id="some/upstream-model",
        adapter_version="2.1.0",
        confidence=0.88,
        latency_ms=300,
        timestamp_unix=int(time.time()),
    )
    ir = _make_ir()
    ir.provenance.append(prior)
    out = FinbertAdapter().egress(_mock_output(), ir, latency_ms=100)
    assert out.provenance[0].model_id == "some/upstream-model"
    assert out.provenance[0].confidence == 0.88
    assert len(out.provenance) == 2
    assert out.provenance[1].model_id == "ProsusAI/finbert"


def test_egress_does_not_mutate_original_ir() -> None:
    ir = _make_ir(content="Earnings beat estimates by 8%.")
    original_content = ir.payload.content
    FinbertAdapter().egress(_mock_output(), ir, latency_ms=100)
    assert ir.payload.content == original_content
    assert ir.payload.labels is None


# ---------------------------------------------------------------------------
# GROUP D — Edge cases (defensive egress, no raise)
# ---------------------------------------------------------------------------

def test_egress_empty_list_output_no_crash() -> None:
    out = FinbertAdapter().egress([], _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)


def test_egress_empty_list_output_labels_empty() -> None:
    out = FinbertAdapter().egress([], _make_ir(), latency_ms=0)
    assert out.payload.labels == []


def test_egress_empty_list_output_confidence_zero() -> None:
    out = FinbertAdapter().egress([], _make_ir(), latency_ms=0)
    assert out.provenance[-1].confidence == 0.0


def test_egress_none_output_no_crash() -> None:
    out = FinbertAdapter().egress(None, _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)


def test_egress_none_output_labels_empty() -> None:
    out = FinbertAdapter().egress(None, _make_ir(), latency_ms=0)
    assert out.payload.labels == []


def test_egress_dict_output_no_crash() -> None:
    out = FinbertAdapter().egress({"label": "positive", "score": 0.9}, _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)


def test_egress_dict_output_labels_empty() -> None:
    out = FinbertAdapter().egress({"label": "positive", "score": 0.9}, _make_ir(), latency_ms=0)
    assert out.payload.labels == []


def test_egress_missing_label_key_no_crash() -> None:
    out = FinbertAdapter().egress([{"score": 0.75}], _make_ir(), latency_ms=50)
    assert isinstance(out, CanonicalIR)


def test_egress_missing_score_key_no_crash() -> None:
    out = FinbertAdapter().egress([{"label": "positive"}], _make_ir(), latency_ms=50)
    assert isinstance(out, CanonicalIR)


def test_egress_score_zero_confidence_zero_in_provenance() -> None:
    out = FinbertAdapter().egress(_mock_output("neutral", 0.0), _make_ir(), latency_ms=100)
    assert out.provenance[-1].confidence == 0.0


def test_egress_latency_ms_zero_valid_provenance_entry() -> None:
    out = FinbertAdapter().egress(_mock_output(), _make_ir(), latency_ms=0)
    assert out.provenance[-1].latency_ms == 0


def test_egress_provenance_always_appended_on_empty_output() -> None:
    ir = _make_ir()
    out = FinbertAdapter().egress([], ir, latency_ms=0)
    assert len(out.provenance) == len(ir.provenance) + 1


def test_egress_string_output_no_crash() -> None:
    out = FinbertAdapter().egress("positive", _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)


# ---------------------------------------------------------------------------
# GROUP E — AdapterValidator direct pass (non-parametrized)
# ---------------------------------------------------------------------------

def test_validator_passes() -> None:
    AdapterValidator(FinbertAdapter()).assert_valid()


def test_validator_result_has_no_errors() -> None:
    result = AdapterValidator(FinbertAdapter(), fixtures=ALL_FIXTURES).run()
    assert result.passed, result.summary()
