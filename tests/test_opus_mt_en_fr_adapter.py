"""Tests for OpusMtEnFrAdapter — unit behaviour and full AdapterValidator suite."""

from __future__ import annotations

import time
import uuid
from typing import Any

import pytest

from opus_mt_en_fr_adapter import OpusMtEnFrAdapter
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
# IR factory
# ---------------------------------------------------------------------------

def _make_ir(
    content: str | None = "How are you?",
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
            task_type=TaskType.translate,
            domain=Domain.multilingual,
            priority=2,
            latency_budget_ms=2000,
        ),
        payload=Payload(**payload_kwargs),
        compliance_envelope=compliance or ComplianceEnvelope(),
    )


# ---------------------------------------------------------------------------
# Mock pipeline output factory
# ---------------------------------------------------------------------------

def _make_output(
    translation: str = "Comment allez-vous ?",
) -> list[dict[str, Any]]:
    return [{"translation_text": translation}]


# ---------------------------------------------------------------------------
# GROUP A — Validator fixture suite (all standard fixtures, parametrized)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", ALL_FIXTURES)
def test_all_fixtures(fixture: CanonicalIR) -> None:
    AdapterValidator(OpusMtEnFrAdapter()).assert_valid_on(fixture)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_model_id() -> None:
    assert OpusMtEnFrAdapter().MODEL_ID == "Helsinki-NLP/opus-mt-en-fr"


def test_adapter_version_semver() -> None:
    ver = OpusMtEnFrAdapter().ADAPTER_VERSION
    parts = ver.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_adapter_version_is_1_0_0() -> None:
    assert OpusMtEnFrAdapter().ADAPTER_VERSION == "1.0.0"


# ---------------------------------------------------------------------------
# GROUP C — Ingress correctness
# ---------------------------------------------------------------------------

def test_ingress_returns_dict() -> None:
    result = OpusMtEnFrAdapter().ingress(_make_ir())
    assert isinstance(result, dict)


def test_ingress_has_text_key() -> None:
    result = OpusMtEnFrAdapter().ingress(_make_ir())
    assert "text" in result


def test_ingress_text_matches_payload_content() -> None:
    ir = _make_ir(content="Good morning, how are you?")
    result = OpusMtEnFrAdapter().ingress(ir)
    assert result["text"] == "Good morning, how are you?"


def test_ingress_empty_string_content_returns_empty_text() -> None:
    ir = _make_ir(content="")
    result = OpusMtEnFrAdapter().ingress(ir)
    assert result["text"] == ""


def test_ingress_none_content_returns_empty_text() -> None:
    ir = _make_ir(content=None, modality="structured")
    result = OpusMtEnFrAdapter().ingress(ir)
    assert result["text"] == ""


def test_ingress_multi_sentence_content_passes_through() -> None:
    text = "The cat sat on the mat. The dog ran in the park. It was a sunny day."
    ir = _make_ir(content=text)
    result = OpusMtEnFrAdapter().ingress(ir)
    assert result["text"] == text


def test_ingress_never_returns_none() -> None:
    adapter = OpusMtEnFrAdapter()
    assert adapter.ingress(_make_ir()) is not None
    assert adapter.ingress(_make_ir(content=None, modality="structured")) is not None


def test_ingress_only_has_text_key() -> None:
    result = OpusMtEnFrAdapter().ingress(_make_ir())
    assert list(result.keys()) == ["text"]


# ---------------------------------------------------------------------------
# GROUP B — Egress correctness with mock output
# ---------------------------------------------------------------------------

def test_egress_returns_canonical_ir() -> None:
    out = OpusMtEnFrAdapter().egress(_make_output(), _make_ir(), latency_ms=150)
    assert isinstance(out, CanonicalIR)


def test_egress_translation_text_stored_in_payload_content() -> None:
    out = OpusMtEnFrAdapter().egress(
        _make_output("Comment allez-vous ?"), _make_ir(), latency_ms=150
    )
    assert out.payload.content == "Comment allez-vous ?"


def test_egress_payload_content_equals_exact_mock_string() -> None:
    translation = "Bonjour, comment allez-vous aujourd'hui ?"
    out = OpusMtEnFrAdapter().egress(_make_output(translation), _make_ir(), latency_ms=200)
    assert out.payload.content == translation


def test_egress_appends_exactly_one_provenance_entry() -> None:
    ir = _make_ir()
    out = OpusMtEnFrAdapter().egress(_make_output(), ir, latency_ms=150)
    assert len(out.provenance) == len(ir.provenance) + 1


def test_egress_provenance_confidence_is_1_0() -> None:
    out = OpusMtEnFrAdapter().egress(_make_output(), _make_ir(), latency_ms=150)
    assert out.provenance[-1].confidence == 1.0


def test_egress_provenance_confidence_is_not_0_0() -> None:
    out = OpusMtEnFrAdapter().egress(_make_output(), _make_ir(), latency_ms=150)
    assert out.provenance[-1].confidence != 0.0


def test_egress_provenance_model_id_matches() -> None:
    out = OpusMtEnFrAdapter().egress(_make_output(), _make_ir(), latency_ms=150)
    assert out.provenance[-1].model_id == "Helsinki-NLP/opus-mt-en-fr"


def test_egress_task_header_carried_forward_unchanged() -> None:
    ir = _make_ir()
    out = OpusMtEnFrAdapter().egress(_make_output(), ir, latency_ms=150)
    assert out.task_header.model_dump() == ir.task_header.model_dump()


def test_egress_compliance_envelope_carried_forward_unchanged() -> None:
    compliance = ComplianceEnvelope(
        required_tags=["gdpr"],
        pii_present=False,
        retention_policy="3y",
        data_residency=["eu-west-1"],
        purpose_limitation="translation",
    )
    ir = _make_ir(compliance=compliance)
    out = OpusMtEnFrAdapter().egress(_make_output(), ir, latency_ms=150)
    assert out.compliance_envelope.model_dump() == compliance.model_dump()


def test_egress_pii_present_remains_false() -> None:
    ir = _make_ir(compliance=ComplianceEnvelope(pii_present=False))
    out = OpusMtEnFrAdapter().egress(_make_output(), ir, latency_ms=150)
    assert out.compliance_envelope.pii_present is False


def test_egress_pii_present_not_set_true() -> None:
    ir = _make_ir()
    out = OpusMtEnFrAdapter().egress(_make_output(), ir, latency_ms=150)
    assert out.compliance_envelope.pii_present is not True


def test_egress_payload_entities_not_modified_when_none() -> None:
    ir = _make_ir()
    assert ir.payload.entities is None
    out = OpusMtEnFrAdapter().egress(_make_output(), ir, latency_ms=150)
    assert out.payload.entities is None


def test_egress_payload_labels_not_modified_when_none() -> None:
    ir = _make_ir()
    assert ir.payload.labels is None
    out = OpusMtEnFrAdapter().egress(_make_output(), ir, latency_ms=150)
    assert out.payload.labels is None


def test_egress_latency_ms_stored_in_provenance() -> None:
    out = OpusMtEnFrAdapter().egress(_make_output(), _make_ir(), latency_ms=333)
    assert out.provenance[-1].latency_ms == 333


def test_egress_realistic_french_translation_stored_correctly() -> None:
    translation = "Je voudrais un café, s'il vous plaît."
    out = OpusMtEnFrAdapter().egress(_make_output(translation), _make_ir(), latency_ms=100)
    assert out.payload.content == translation


def test_egress_another_french_translation_stored_correctly() -> None:
    translation = "Le chat est sur le tapis."
    out = OpusMtEnFrAdapter().egress(_make_output(translation), _make_ir(), latency_ms=120)
    assert out.payload.content == translation


def test_egress_unicode_accented_characters_stored_correctly() -> None:
    translation = "très bien"
    out = OpusMtEnFrAdapter().egress(_make_output(translation), _make_ir(), latency_ms=90)
    assert out.payload.content == "très bien"


def test_egress_empty_translation_string_stored_correctly() -> None:
    out = OpusMtEnFrAdapter().egress(_make_output(""), _make_ir(), latency_ms=0)
    assert out.payload.content == ""


def test_egress_existing_provenance_not_modified() -> None:
    prior = ProvenanceEntry(
        model_id="some/prior-model",
        adapter_version="2.1.0",
        confidence=0.75,
        latency_ms=300,
        timestamp_unix=int(time.time()),
    )
    ir = _make_ir()
    ir.provenance.append(prior)

    out = OpusMtEnFrAdapter().egress(_make_output(), ir, latency_ms=150)

    assert out.provenance[0].model_id == "some/prior-model"
    assert out.provenance[0].confidence == 0.75
    assert len(out.provenance) == 2
    assert out.provenance[1].model_id == "Helsinki-NLP/opus-mt-en-fr"


def test_egress_does_not_mutate_original_ir() -> None:
    ir = _make_ir(content="Hello world.")
    original_content = ir.payload.content
    OpusMtEnFrAdapter().egress(_make_output(), ir, latency_ms=150)
    assert ir.payload.content == original_content


# ---------------------------------------------------------------------------
# GROUP D — Edge cases (defensive egress)
# ---------------------------------------------------------------------------

def test_egress_empty_list_sets_content_to_empty_string() -> None:
    out = OpusMtEnFrAdapter().egress([], _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)
    assert out.payload.content == ""


def test_egress_empty_list_does_not_raise() -> None:
    out = OpusMtEnFrAdapter().egress([], _make_ir(), latency_ms=0)
    assert out is not None


def test_egress_none_output_does_not_raise() -> None:
    out = OpusMtEnFrAdapter().egress(None, _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)
    assert out.payload.content == ""


def test_egress_dict_output_does_not_raise() -> None:
    out = OpusMtEnFrAdapter().egress({"translation_text": "bonjour"}, _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)
    assert out.payload.content == ""


def test_egress_missing_translation_text_key_sets_content_to_empty_string() -> None:
    bad_output = [{"other_key": "some value"}]
    out = OpusMtEnFrAdapter().egress(bad_output, _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)
    assert out.payload.content == ""


def test_egress_translation_text_none_in_dict_does_not_raise() -> None:
    bad_output = [{"translation_text": None}]
    out = OpusMtEnFrAdapter().egress(bad_output, _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)
    assert out.payload.content == ""


def test_egress_latency_ms_zero_produces_valid_provenance() -> None:
    out = OpusMtEnFrAdapter().egress(_make_output(), _make_ir(), latency_ms=0)
    assert out.provenance[-1].latency_ms == 0
    assert out.provenance[-1].confidence == 1.0


def test_egress_very_long_translation_stored_correctly() -> None:
    long_translation = "Bonjour. " * 500
    out = OpusMtEnFrAdapter().egress(_make_output(long_translation), _make_ir(), latency_ms=800)
    assert out.payload.content == long_translation


def test_egress_provenance_appended_even_on_empty_output() -> None:
    ir = _make_ir()
    out = OpusMtEnFrAdapter().egress([], ir, latency_ms=0)
    assert len(out.provenance) == len(ir.provenance) + 1
    assert out.provenance[-1].confidence == 1.0


def test_egress_string_output_does_not_raise() -> None:
    out = OpusMtEnFrAdapter().egress("bare string", _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)
    assert out.payload.content == ""


# ---------------------------------------------------------------------------
# GROUP E — AdapterValidator direct pass (non-parametrized)
# ---------------------------------------------------------------------------

def test_validator_passes() -> None:
    AdapterValidator(OpusMtEnFrAdapter()).assert_valid()
