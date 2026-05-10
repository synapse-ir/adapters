"""Tests for TwitterRobertaSentimentAdapter — unit behaviour and full AdapterValidator suite."""

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
from twitter_roberta_sentiment_adapter import TwitterRobertaSentimentAdapter


# ---------------------------------------------------------------------------
# IR factory
# ---------------------------------------------------------------------------

def _make_ir(
    content: str | None = "Covid cases are increasing fast!",
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
            domain=Domain.conversational,
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
    label: str = "Negative",
    score: float = 0.7236,
) -> list[dict[str, Any]]:
    return [{"label": label, "score": score}]


# ---------------------------------------------------------------------------
# GROUP A — Validator fixture suite (all standard fixtures, parametrized)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", ALL_FIXTURES)
def test_all_fixtures(fixture: CanonicalIR) -> None:
    AdapterValidator(TwitterRobertaSentimentAdapter()).assert_valid_on(fixture)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_model_id() -> None:
    assert TwitterRobertaSentimentAdapter().MODEL_ID == "cardiffnlp/twitter-roberta-base-sentiment-latest"


def test_adapter_version_semver() -> None:
    ver = TwitterRobertaSentimentAdapter().ADAPTER_VERSION
    parts = ver.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_adapter_version_is_1_0_0() -> None:
    assert TwitterRobertaSentimentAdapter().ADAPTER_VERSION == "1.0.0"


# ---------------------------------------------------------------------------
# GROUP C — Ingress correctness
# ---------------------------------------------------------------------------

def test_ingress_returns_dict() -> None:
    result = TwitterRobertaSentimentAdapter().ingress(_make_ir())
    assert isinstance(result, dict)


def test_ingress_non_empty_content_stored_as_text() -> None:
    ir = _make_ir(content="This is great news!")
    result = TwitterRobertaSentimentAdapter().ingress(ir)
    assert result["text"] == "This is great news!"


def test_ingress_empty_string_returns_empty_text() -> None:
    ir = _make_ir(content="")
    result = TwitterRobertaSentimentAdapter().ingress(ir)
    assert result["text"] == ""


def test_ingress_none_content_returns_empty_text() -> None:
    ir = _make_ir(content=None, modality="structured")
    result = TwitterRobertaSentimentAdapter().ingress(ir)
    assert result["text"] == ""


def test_ingress_has_text_key() -> None:
    result = TwitterRobertaSentimentAdapter().ingress(_make_ir())
    assert "text" in result


def test_ingress_never_returns_none() -> None:
    adapter = TwitterRobertaSentimentAdapter()
    assert adapter.ingress(_make_ir()) is not None
    assert adapter.ingress(_make_ir(content=None, modality="structured")) is not None


# ---------------------------------------------------------------------------
# GROUP B — Egress correctness with mock output
# ---------------------------------------------------------------------------

def test_egress_returns_canonical_ir() -> None:
    out = TwitterRobertaSentimentAdapter().egress(_mock_output(), _make_ir(), latency_ms=100)
    assert isinstance(out, CanonicalIR)


def test_egress_negative_label_stored_correctly() -> None:
    out = TwitterRobertaSentimentAdapter().egress(
        _mock_output("Negative", 0.7236), _make_ir(), latency_ms=100
    )
    assert out.payload.labels[0].label == "Negative"  # type: ignore[index]


def test_egress_neutral_label_stored_correctly() -> None:
    out = TwitterRobertaSentimentAdapter().egress(
        _mock_output("Neutral", 0.5412), _make_ir(), latency_ms=100
    )
    assert out.payload.labels[0].label == "Neutral"  # type: ignore[index]


def test_egress_positive_label_stored_correctly() -> None:
    out = TwitterRobertaSentimentAdapter().egress(
        _mock_output("Positive", 0.9123), _make_ir(), latency_ms=100
    )
    assert out.payload.labels[0].label == "Positive"  # type: ignore[index]


def test_egress_labels_are_title_cased() -> None:
    for label in ("Negative", "Neutral", "Positive"):
        out = TwitterRobertaSentimentAdapter().egress(
            _mock_output(label, 0.8), _make_ir(), latency_ms=100
        )
        stored = out.payload.labels[0].label  # type: ignore[index]
        assert stored == label
        assert stored[0].isupper()


def test_egress_score_stored_correctly_in_payload_labels() -> None:
    out = TwitterRobertaSentimentAdapter().egress(
        _mock_output("Negative", 0.7236), _make_ir(), latency_ms=100
    )
    assert out.payload.labels[0].score == pytest.approx(0.7236, abs=1e-6)  # type: ignore[index]


def test_egress_confidence_in_provenance_equals_score() -> None:
    out = TwitterRobertaSentimentAdapter().egress(
        _mock_output("Positive", 0.9123), _make_ir(), latency_ms=100
    )
    assert out.provenance[-1].confidence == pytest.approx(0.9123, abs=1e-6)


def test_egress_provenance_length_is_original_plus_one() -> None:
    ir = _make_ir()
    out = TwitterRobertaSentimentAdapter().egress(_mock_output(), ir, latency_ms=100)
    assert len(out.provenance) == len(ir.provenance) + 1


def test_egress_task_header_unchanged() -> None:
    ir = _make_ir()
    out = TwitterRobertaSentimentAdapter().egress(_mock_output(), ir, latency_ms=100)
    assert out.task_header.model_dump() == ir.task_header.model_dump()


def test_egress_compliance_envelope_unchanged() -> None:
    compliance = ComplianceEnvelope(
        required_tags=["gdpr"],
        pii_present=False,
        retention_policy="30d",
    )
    ir = _make_ir(compliance=compliance)
    out = TwitterRobertaSentimentAdapter().egress(_mock_output(), ir, latency_ms=100)
    assert out.compliance_envelope.model_dump() == compliance.model_dump()


def test_egress_pii_present_remains_false() -> None:
    ir = _make_ir(compliance=ComplianceEnvelope(pii_present=False))
    out = TwitterRobertaSentimentAdapter().egress(_mock_output(), ir, latency_ms=100)
    assert out.compliance_envelope.pii_present is False


def test_egress_pii_present_never_set_true() -> None:
    ir = _make_ir()
    out = TwitterRobertaSentimentAdapter().egress(_mock_output(), ir, latency_ms=100)
    assert out.compliance_envelope.pii_present is not True


def test_egress_payload_labels_is_list_of_length_one() -> None:
    out = TwitterRobertaSentimentAdapter().egress(_mock_output(), _make_ir(), latency_ms=100)
    assert isinstance(out.payload.labels, list)
    assert len(out.payload.labels) == 1


def test_egress_payload_labels_item_is_classification() -> None:
    out = TwitterRobertaSentimentAdapter().egress(_mock_output(), _make_ir(), latency_ms=100)
    assert isinstance(out.payload.labels[0], Classification)  # type: ignore[index]


def test_egress_payload_labels_first_label_matches_mock() -> None:
    out = TwitterRobertaSentimentAdapter().egress(
        _mock_output("Neutral", 0.55), _make_ir(), latency_ms=100
    )
    assert out.payload.labels[0].label == "Neutral"  # type: ignore[index]


def test_egress_payload_labels_first_score_matches_mock() -> None:
    out = TwitterRobertaSentimentAdapter().egress(
        _mock_output("Positive", 0.8812), _make_ir(), latency_ms=100
    )
    assert out.payload.labels[0].score == pytest.approx(0.8812, abs=1e-6)  # type: ignore[index]


def test_egress_latency_ms_stored_in_provenance() -> None:
    out = TwitterRobertaSentimentAdapter().egress(_mock_output(), _make_ir(), latency_ms=250)
    assert out.provenance[-1].latency_ms == 250


def test_egress_provenance_model_id_matches() -> None:
    out = TwitterRobertaSentimentAdapter().egress(_mock_output(), _make_ir(), latency_ms=100)
    assert out.provenance[-1].model_id == "cardiffnlp/twitter-roberta-base-sentiment-latest"


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

    out = TwitterRobertaSentimentAdapter().egress(_mock_output(), ir, latency_ms=100)

    assert out.provenance[0].model_id == "some/prior-model"
    assert out.provenance[0].confidence == 0.72
    assert len(out.provenance) == 2
    assert out.provenance[1].model_id == "cardiffnlp/twitter-roberta-base-sentiment-latest"


def test_egress_does_not_mutate_original_ir() -> None:
    ir = _make_ir(content="Original tweet text.")
    original_content = ir.payload.content
    TwitterRobertaSentimentAdapter().egress(_mock_output(), ir, latency_ms=100)
    assert ir.payload.content == original_content
    assert ir.payload.labels is None


def test_egress_empty_compliance_envelope_carried_forward() -> None:
    ir = _make_ir(compliance=ComplianceEnvelope())
    out = TwitterRobertaSentimentAdapter().egress(_mock_output(), ir, latency_ms=100)
    assert out.compliance_envelope.model_dump() == ComplianceEnvelope().model_dump()


# ---------------------------------------------------------------------------
# GROUP D — Edge cases (defensive egress)
# ---------------------------------------------------------------------------

def test_egress_empty_list_output_no_crash() -> None:
    out = TwitterRobertaSentimentAdapter().egress([], _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)


def test_egress_empty_list_output_labels_empty() -> None:
    out = TwitterRobertaSentimentAdapter().egress([], _make_ir(), latency_ms=0)
    assert out.payload.labels == []


def test_egress_empty_list_output_confidence_zero() -> None:
    out = TwitterRobertaSentimentAdapter().egress([], _make_ir(), latency_ms=0)
    assert out.provenance[-1].confidence == 0.0


def test_egress_none_output_no_crash() -> None:
    out = TwitterRobertaSentimentAdapter().egress(None, _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)


def test_egress_none_output_labels_empty() -> None:
    out = TwitterRobertaSentimentAdapter().egress(None, _make_ir(), latency_ms=0)
    assert out.payload.labels == []


def test_egress_dict_instead_of_list_no_crash() -> None:
    out = TwitterRobertaSentimentAdapter().egress({"label": "Negative", "score": 0.7}, _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)


def test_egress_dict_instead_of_list_labels_empty() -> None:
    out = TwitterRobertaSentimentAdapter().egress({"label": "Negative", "score": 0.7}, _make_ir(), latency_ms=0)
    assert out.payload.labels == []


def test_egress_missing_label_key_no_crash() -> None:
    out = TwitterRobertaSentimentAdapter().egress([{"score": 0.7}], _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)


def test_egress_missing_label_key_stores_empty_label_string() -> None:
    out = TwitterRobertaSentimentAdapter().egress([{"score": 0.7}], _make_ir(), latency_ms=0)
    assert out.payload.labels[0].label == ""  # type: ignore[index]


def test_egress_missing_score_key_no_crash() -> None:
    out = TwitterRobertaSentimentAdapter().egress([{"label": "Positive"}], _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)


def test_egress_missing_score_key_confidence_zero() -> None:
    out = TwitterRobertaSentimentAdapter().egress([{"label": "Positive"}], _make_ir(), latency_ms=0)
    assert out.provenance[-1].confidence == 0.0


def test_egress_score_zero_confidence_is_zero_not_error() -> None:
    out = TwitterRobertaSentimentAdapter().egress(
        _mock_output("Neutral", 0.0), _make_ir(), latency_ms=0
    )
    assert isinstance(out, CanonicalIR)
    assert out.provenance[-1].confidence == 0.0


def test_egress_latency_ms_zero_produces_valid_provenance() -> None:
    out = TwitterRobertaSentimentAdapter().egress(_mock_output(), _make_ir(), latency_ms=0)
    assert out.provenance[-1].latency_ms == 0
    assert 0.0 <= out.provenance[-1].confidence <= 1.0


def test_egress_provenance_always_appended_even_on_bad_output() -> None:
    ir = _make_ir()
    out = TwitterRobertaSentimentAdapter().egress(None, ir, latency_ms=0)
    assert len(out.provenance) == len(ir.provenance) + 1


def test_egress_string_output_no_crash() -> None:
    out = TwitterRobertaSentimentAdapter().egress("Negative", _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)


# ---------------------------------------------------------------------------
# GROUP E — AdapterValidator direct pass (non-parametrized)
# ---------------------------------------------------------------------------

def test_validator_passes() -> None:
    AdapterValidator(TwitterRobertaSentimentAdapter()).assert_valid()
