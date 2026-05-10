"""Tests for MsMarcoCrossEncoderAdapter — unit behaviour and full AdapterValidator suite."""

from __future__ import annotations

import math
import time
import uuid
from typing import Any

import numpy
import pytest

from ms_marco_cross_encoder_adapter import MsMarcoCrossEncoderAdapter
from synapse_sdk.testing import AdapterValidator
from synapse_sdk.testing.fixtures import ALL_FIXTURES
from synapse_sdk.types import (
    CanonicalIR,
    ComplianceEnvelope,
    Domain,
    Payload,
    ProvenanceEntry,
    TaskHeader,
    TaskType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0


def _make_ir(
    content: str | None = "Berlin had a population of 3.7 million in 2022.",
    query: str | None = "How many people live in Berlin?",
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
            task_type=TaskType.rank,
            domain=Domain.general,
            priority=2,
            latency_budget_ms=800,
            query=query,
        ),
        payload=Payload(**payload_kwargs),
        compliance_envelope=compliance or ComplianceEnvelope(),
    )


def _make_numpy_output(raw_score: float = 8.607138) -> numpy.ndarray:
    return numpy.array([raw_score], dtype=numpy.float32)


# ---------------------------------------------------------------------------
# GROUP A — Validator fixture suite (all standard fixtures, parametrized)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", ALL_FIXTURES)
def test_all_fixtures(fixture: CanonicalIR) -> None:
    AdapterValidator(MsMarcoCrossEncoderAdapter()).assert_valid_on(fixture)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_model_id() -> None:
    assert MsMarcoCrossEncoderAdapter().MODEL_ID == "cross-encoder/ms-marco-MiniLM-L6-v2"


def test_adapter_version_semver() -> None:
    ver = MsMarcoCrossEncoderAdapter().ADAPTER_VERSION
    parts = ver.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_adapter_version_is_1_0_0() -> None:
    assert MsMarcoCrossEncoderAdapter().ADAPTER_VERSION == "1.0.0"


# ---------------------------------------------------------------------------
# GROUP C — Ingress correctness
# ---------------------------------------------------------------------------

def test_ingress_returns_dict() -> None:
    result = MsMarcoCrossEncoderAdapter().ingress(_make_ir())
    assert isinstance(result, dict)


def test_ingress_has_query_and_passage_keys() -> None:
    result = MsMarcoCrossEncoderAdapter().ingress(_make_ir())
    assert "query" in result
    assert "passage" in result


def test_ingress_query_and_passage_present() -> None:
    ir = _make_ir(
        query="How many people live in Berlin?",
        content="Berlin had a population of 3.7 million in 2022.",
    )
    result = MsMarcoCrossEncoderAdapter().ingress(ir)
    assert result["query"] == "How many people live in Berlin?"
    assert result["passage"] == "Berlin had a population of 3.7 million in 2022."


def test_ingress_none_query_returns_empty_string() -> None:
    ir = _make_ir(query=None)
    result = MsMarcoCrossEncoderAdapter().ingress(ir)
    assert result["query"] == ""


def test_ingress_none_content_returns_empty_passage() -> None:
    ir = _make_ir(content=None, modality="structured")
    result = MsMarcoCrossEncoderAdapter().ingress(ir)
    assert result["passage"] == ""


def test_ingress_both_none_returns_empty_strings() -> None:
    ir = _make_ir(query=None, content=None, modality="structured")
    result = MsMarcoCrossEncoderAdapter().ingress(ir)
    assert result["query"] == ""
    assert result["passage"] == ""


def test_ingress_non_empty_query_stored_under_query_key() -> None:
    ir = _make_ir(query="What is the capital of France?")
    result = MsMarcoCrossEncoderAdapter().ingress(ir)
    assert result["query"] == "What is the capital of France?"


def test_ingress_passage_stored_under_passage_key() -> None:
    ir = _make_ir(content="Paris is the capital and most populous city of France.")
    result = MsMarcoCrossEncoderAdapter().ingress(ir)
    assert result["passage"] == "Paris is the capital and most populous city of France."


def test_ingress_never_returns_none() -> None:
    adapter = MsMarcoCrossEncoderAdapter()
    assert adapter.ingress(_make_ir()) is not None
    assert adapter.ingress(_make_ir(query=None, content=None, modality="structured")) is not None


# ---------------------------------------------------------------------------
# GROUP B — Egress correctness with mock output
# ---------------------------------------------------------------------------

def test_egress_returns_canonical_ir() -> None:
    out = MsMarcoCrossEncoderAdapter().egress(_make_numpy_output(), _make_ir(), latency_ms=50)
    assert isinstance(out, CanonicalIR)


def test_egress_numpy_array_input_sigmoid_normalized() -> None:
    raw = 8.607138
    out = MsMarcoCrossEncoderAdapter().egress(_make_numpy_output(raw), _make_ir(), latency_ms=50)
    expected = _sigmoid(float(numpy.float32(raw)))
    assert out.payload.score == pytest.approx(expected, abs=1e-6)


def test_egress_plain_float_input_sigmoid_normalized() -> None:
    raw = 8.607138
    out = MsMarcoCrossEncoderAdapter().egress(raw, _make_ir(), latency_ms=50)
    expected = _sigmoid(raw)
    assert out.payload.score == pytest.approx(expected, abs=1e-6)


def test_egress_positive_raw_score_normalized_above_0_5() -> None:
    out = MsMarcoCrossEncoderAdapter().egress(3.0, _make_ir(), latency_ms=50)
    assert out.payload.score > 0.5  # type: ignore[operator]


def test_egress_negative_raw_score_normalized_below_0_5() -> None:
    out = MsMarcoCrossEncoderAdapter().egress(-3.0, _make_ir(), latency_ms=50)
    assert out.payload.score < 0.5  # type: ignore[operator]


def test_egress_raw_score_zero_normalized_exactly_0_5() -> None:
    out = MsMarcoCrossEncoderAdapter().egress(0.0, _make_ir(), latency_ms=50)
    assert out.payload.score == pytest.approx(0.5, abs=1e-10)


def test_egress_very_large_positive_raw_score_near_1_0() -> None:
    out = MsMarcoCrossEncoderAdapter().egress(10.0, _make_ir(), latency_ms=50)
    assert out.payload.score > 0.99  # type: ignore[operator]


def test_egress_very_large_negative_raw_score_near_0_0() -> None:
    out = MsMarcoCrossEncoderAdapter().egress(-10.0, _make_ir(), latency_ms=50)
    assert out.payload.score < 0.01  # type: ignore[operator]


def test_egress_score_stored_in_payload_score() -> None:
    out = MsMarcoCrossEncoderAdapter().egress(2.0, _make_ir(), latency_ms=50)
    assert out.payload.score is not None
    assert isinstance(out.payload.score, float)


def test_egress_confidence_in_provenance_equals_normalized_score() -> None:
    out = MsMarcoCrossEncoderAdapter().egress(5.0, _make_ir(), latency_ms=50)
    assert out.provenance[-1].confidence == pytest.approx(out.payload.score, abs=1e-10)  # type: ignore[arg-type]


def test_egress_provenance_list_length_original_plus_one() -> None:
    ir = _make_ir()
    out = MsMarcoCrossEncoderAdapter().egress(3.5, ir, latency_ms=50)
    assert len(out.provenance) == len(ir.provenance) + 1


def test_egress_task_header_carried_forward_unchanged() -> None:
    ir = _make_ir()
    out = MsMarcoCrossEncoderAdapter().egress(2.0, ir, latency_ms=50)
    assert out.task_header.model_dump() == ir.task_header.model_dump()


def test_egress_compliance_envelope_carried_forward_unchanged() -> None:
    compliance = ComplianceEnvelope(
        required_tags=["gdpr"],
        pii_present=False,
        retention_policy="1y",
    )
    ir = _make_ir(compliance=compliance)
    out = MsMarcoCrossEncoderAdapter().egress(2.0, ir, latency_ms=50)
    assert out.compliance_envelope.model_dump() == compliance.model_dump()


def test_egress_pii_present_remains_false() -> None:
    ir = _make_ir(compliance=ComplianceEnvelope(pii_present=False))
    out = MsMarcoCrossEncoderAdapter().egress(2.0, ir, latency_ms=50)
    assert out.compliance_envelope.pii_present is False


def test_egress_pii_present_not_set_true() -> None:
    ir = _make_ir()
    out = MsMarcoCrossEncoderAdapter().egress(2.0, ir, latency_ms=50)
    assert out.compliance_envelope.pii_present is not True


def test_egress_payload_content_not_modified() -> None:
    original_content = "Berlin had a population of 3.7 million in 2022."
    ir = _make_ir(content=original_content)
    out = MsMarcoCrossEncoderAdapter().egress(2.0, ir, latency_ms=50)
    assert out.payload.content == original_content


def test_egress_payload_entities_not_modified() -> None:
    ir = _make_ir()
    assert ir.payload.entities is None
    out = MsMarcoCrossEncoderAdapter().egress(2.0, ir, latency_ms=50)
    assert out.payload.entities is None


def test_egress_payload_labels_not_modified() -> None:
    ir = _make_ir()
    assert ir.payload.labels is None
    out = MsMarcoCrossEncoderAdapter().egress(2.0, ir, latency_ms=50)
    assert out.payload.labels is None


def test_egress_latency_ms_stored_in_provenance() -> None:
    out = MsMarcoCrossEncoderAdapter().egress(2.0, _make_ir(), latency_ms=333)
    assert out.provenance[-1].latency_ms == 333


def test_egress_provenance_model_id_matches() -> None:
    out = MsMarcoCrossEncoderAdapter().egress(2.0, _make_ir(), latency_ms=50)
    assert out.provenance[-1].model_id == "cross-encoder/ms-marco-MiniLM-L6-v2"


def test_egress_existing_provenance_not_modified() -> None:
    prior = ProvenanceEntry(
        model_id="some/prior-model",
        adapter_version="1.0.0",
        confidence=0.80,
        latency_ms=200,
        timestamp_unix=int(time.time()),
    )
    ir = _make_ir()
    ir.provenance.append(prior)

    out = MsMarcoCrossEncoderAdapter().egress(2.0, ir, latency_ms=50)

    assert out.provenance[0].model_id == "some/prior-model"
    assert out.provenance[0].confidence == 0.80
    assert len(out.provenance) == 2
    assert out.provenance[1].model_id == "cross-encoder/ms-marco-MiniLM-L6-v2"


def test_egress_does_not_mutate_original_ir() -> None:
    ir = _make_ir(content="Original passage.")
    original_content = ir.payload.content
    MsMarcoCrossEncoderAdapter().egress(2.0, ir, latency_ms=50)
    assert ir.payload.content == original_content
    assert ir.payload.score is None


def test_egress_numpy_array_score_matches_expected_sigmoid() -> None:
    raw = 2.5
    arr = numpy.array([raw], dtype=numpy.float32)
    out = MsMarcoCrossEncoderAdapter().egress(arr, _make_ir(), latency_ms=50)
    expected = _sigmoid(float(numpy.float32(raw)))
    assert out.payload.score == pytest.approx(expected, abs=1e-5)


def test_egress_int_raw_score_sigmoid_applied() -> None:
    out = MsMarcoCrossEncoderAdapter().egress(5, _make_ir(), latency_ms=50)
    expected = _sigmoid(5.0)
    assert out.payload.score == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# GROUP D — Edge cases (defensive egress)
# ---------------------------------------------------------------------------

def test_egress_none_output_no_crash() -> None:
    out = MsMarcoCrossEncoderAdapter().egress(None, _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)


def test_egress_none_output_score_is_sigmoid_zero() -> None:
    out = MsMarcoCrossEncoderAdapter().egress(None, _make_ir(), latency_ms=0)
    assert out.payload.score == pytest.approx(0.5, abs=1e-10)


def test_egress_empty_numpy_array_no_crash() -> None:
    out = MsMarcoCrossEncoderAdapter().egress(numpy.array([]), _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)


def test_egress_empty_numpy_array_raw_score_zero() -> None:
    out = MsMarcoCrossEncoderAdapter().egress(numpy.array([]), _make_ir(), latency_ms=0)
    assert out.payload.score == pytest.approx(0.5, abs=1e-10)


def test_egress_string_output_no_crash() -> None:
    out = MsMarcoCrossEncoderAdapter().egress("8.607138", _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)


def test_egress_string_output_treated_as_unparseable() -> None:
    out = MsMarcoCrossEncoderAdapter().egress("8.607138", _make_ir(), latency_ms=0)
    assert out.payload.score == pytest.approx(0.5, abs=1e-10)


def test_egress_dict_output_no_crash() -> None:
    out = MsMarcoCrossEncoderAdapter().egress({"result": "test"}, _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)


def test_egress_dict_output_treated_as_unparseable() -> None:
    out = MsMarcoCrossEncoderAdapter().egress({"result": "test"}, _make_ir(), latency_ms=0)
    assert out.payload.score == pytest.approx(0.5, abs=1e-10)


def test_egress_int_output_treated_as_float() -> None:
    out = MsMarcoCrossEncoderAdapter().egress(5, _make_ir(), latency_ms=50)
    expected = _sigmoid(5.0)
    assert out.payload.score == pytest.approx(expected, abs=1e-10)


def test_egress_score_always_clamped_to_0_1_lower() -> None:
    out = MsMarcoCrossEncoderAdapter().egress(-1000.0, _make_ir(), latency_ms=0)
    assert 0.0 <= out.payload.score <= 1.0  # type: ignore[operator]


def test_egress_score_always_clamped_to_0_1_upper() -> None:
    out = MsMarcoCrossEncoderAdapter().egress(1000.0, _make_ir(), latency_ms=0)
    assert 0.0 <= out.payload.score <= 1.0  # type: ignore[operator]


def test_egress_latency_ms_zero_valid_provenance_entry() -> None:
    out = MsMarcoCrossEncoderAdapter().egress(2.0, _make_ir(), latency_ms=0)
    assert out.provenance[-1].latency_ms == 0
    assert 0.0 <= out.provenance[-1].confidence <= 1.0


def test_egress_provenance_always_appended_even_on_bad_output() -> None:
    ir = _make_ir()
    out = MsMarcoCrossEncoderAdapter().egress(None, ir, latency_ms=0)
    assert len(out.provenance) == len(ir.provenance) + 1


# ---------------------------------------------------------------------------
# GROUP E — AdapterValidator direct pass (non-parametrized)
# ---------------------------------------------------------------------------

def test_validator_passes() -> None:
    AdapterValidator(MsMarcoCrossEncoderAdapter()).assert_valid()
