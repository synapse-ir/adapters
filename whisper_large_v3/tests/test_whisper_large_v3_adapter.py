"""Tests for WhisperLargeV3Adapter — unit behaviour and full AdapterValidator suite."""

from __future__ import annotations

import time
import uuid
from typing import Any

import pytest

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
from whisper_large_v3_adapter import WhisperLargeV3Adapter


# ---------------------------------------------------------------------------
# IR factory
# ---------------------------------------------------------------------------

def _make_ir(
    content: str | None = "/data/earnings_call.mp3",
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
            task_type=TaskType.generate,
            domain=Domain.multilingual,
            priority=2,
            latency_budget_ms=5000,
        ),
        payload=Payload(**payload_kwargs),
        compliance_envelope=compliance or ComplianceEnvelope(),
    )


# ---------------------------------------------------------------------------
# Mock pipeline output factory
# ---------------------------------------------------------------------------

def _make_output(
    text: str = " And so my fellow Americans...",
) -> dict[str, Any]:
    return {"text": text}


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_model_id() -> None:
    assert WhisperLargeV3Adapter().MODEL_ID == "openai/whisper-large-v3"


def test_adapter_version_semver() -> None:
    ver = WhisperLargeV3Adapter().ADAPTER_VERSION
    parts = ver.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_adapter_version_is_1_0_0() -> None:
    assert WhisperLargeV3Adapter().ADAPTER_VERSION == "1.0.0"


# ---------------------------------------------------------------------------
# GROUP A — Validator fixture suite (all standard fixtures, parametrized)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", ALL_FIXTURES)
def test_all_fixtures(fixture: CanonicalIR) -> None:
    AdapterValidator(WhisperLargeV3Adapter()).assert_valid_on(fixture)


# ---------------------------------------------------------------------------
# GROUP B — Egress correctness (mock output only)
# ---------------------------------------------------------------------------

def test_egress_returns_canonical_ir() -> None:
    out = WhisperLargeV3Adapter().egress(_make_output(), _make_ir(), latency_ms=150)
    assert isinstance(out, CanonicalIR)


def test_egress_transcription_stored_in_payload_content() -> None:
    out = WhisperLargeV3Adapter().egress(
        _make_output(" And so my fellow Americans..."), _make_ir(), latency_ms=150
    )
    assert out.payload.content == "And so my fellow Americans..."


def test_egress_payload_content_equals_stripped_text() -> None:
    mock = _make_output(" Hello world ")
    out = WhisperLargeV3Adapter().egress(mock, _make_ir(), latency_ms=200)
    assert out.payload.content == "Hello world"


def test_egress_leading_whitespace_stripped() -> None:
    mock = _make_output(" The quick brown fox")
    out = WhisperLargeV3Adapter().egress(mock, _make_ir(), latency_ms=100)
    assert out.payload.content == "The quick brown fox"
    assert not out.payload.content.startswith(" ")  # type: ignore[union-attr]


def test_egress_trailing_whitespace_stripped() -> None:
    mock = _make_output("The quick brown fox   ")
    out = WhisperLargeV3Adapter().egress(mock, _make_ir(), latency_ms=100)
    assert out.payload.content == "The quick brown fox"
    assert not out.payload.content.endswith(" ")  # type: ignore[union-attr]


def test_egress_confidence_in_provenance_is_1_0() -> None:
    out = WhisperLargeV3Adapter().egress(_make_output(), _make_ir(), latency_ms=150)
    assert out.provenance[-1].confidence == 1.0


def test_egress_confidence_not_0_0() -> None:
    out = WhisperLargeV3Adapter().egress(_make_output(), _make_ir(), latency_ms=150)
    assert out.provenance[-1].confidence != 0.0


def test_egress_provenance_length_is_original_plus_one() -> None:
    ir = _make_ir()
    out = WhisperLargeV3Adapter().egress(_make_output(), ir, latency_ms=150)
    assert len(out.provenance) == len(ir.provenance) + 1


def test_egress_task_header_carried_forward_unchanged() -> None:
    ir = _make_ir()
    out = WhisperLargeV3Adapter().egress(_make_output(), ir, latency_ms=150)
    assert out.task_header.model_dump() == ir.task_header.model_dump()


def test_egress_compliance_envelope_carried_forward_unchanged() -> None:
    compliance = ComplianceEnvelope(
        required_tags=["sox"],
        pii_present=False,
        retention_policy="7y",
        data_residency=["us-east-1"],
        purpose_limitation="transcription",
    )
    ir = _make_ir(compliance=compliance)
    out = WhisperLargeV3Adapter().egress(_make_output(), ir, latency_ms=150)
    assert out.compliance_envelope.model_dump() == compliance.model_dump()


def test_egress_pii_present_remains_false() -> None:
    ir = _make_ir(compliance=ComplianceEnvelope(pii_present=False))
    out = WhisperLargeV3Adapter().egress(_make_output(), ir, latency_ms=150)
    assert out.compliance_envelope.pii_present is False


def test_egress_pii_present_not_set_true() -> None:
    ir = _make_ir()
    out = WhisperLargeV3Adapter().egress(_make_output(), ir, latency_ms=150)
    assert out.compliance_envelope.pii_present is not True


def test_egress_payload_entities_not_modified() -> None:
    ir = _make_ir()
    assert ir.payload.entities is None
    out = WhisperLargeV3Adapter().egress(_make_output(), ir, latency_ms=150)
    assert out.payload.entities is None


def test_egress_payload_labels_not_modified() -> None:
    ir = _make_ir()
    assert ir.payload.labels is None
    out = WhisperLargeV3Adapter().egress(_make_output(), ir, latency_ms=150)
    assert out.payload.labels is None


def test_egress_latency_ms_stored_correctly_in_provenance() -> None:
    out = WhisperLargeV3Adapter().egress(_make_output(), _make_ir(), latency_ms=777)
    assert out.provenance[-1].latency_ms == 777


def test_egress_empty_string_transcription_stored_correctly() -> None:
    out = WhisperLargeV3Adapter().egress(_make_output(""), _make_ir(), latency_ms=0)
    assert out.payload.content == ""


def test_egress_non_english_transcription_stored_correctly() -> None:
    spanish = "Buenos días, ¿cómo estás?"
    out = WhisperLargeV3Adapter().egress(_make_output(spanish), _make_ir(), latency_ms=200)
    assert out.payload.content == spanish


def test_egress_long_transcription_stored_correctly() -> None:
    long_text = "The market showed significant volatility. " * 300
    out = WhisperLargeV3Adapter().egress(_make_output(long_text), _make_ir(), latency_ms=3000)
    assert out.payload.content == long_text.strip()


def test_egress_does_not_mutate_original_ir() -> None:
    ir = _make_ir(content="/data/audio.mp3")
    original_content = ir.payload.content
    WhisperLargeV3Adapter().egress(_make_output(), ir, latency_ms=150)
    assert ir.payload.content == original_content


def test_egress_existing_provenance_not_modified() -> None:
    prior = ProvenanceEntry(
        model_id="some/prior-model",
        adapter_version="2.0.0",
        confidence=0.85,
        latency_ms=200,
        timestamp_unix=int(time.time()),
    )
    ir = _make_ir()
    ir.provenance.append(prior)

    out = WhisperLargeV3Adapter().egress(_make_output(), ir, latency_ms=150)

    assert out.provenance[0].model_id == "some/prior-model"
    assert out.provenance[0].confidence == 0.85
    assert len(out.provenance) == 2
    assert out.provenance[1].model_id == "openai/whisper-large-v3"


def test_egress_provenance_model_id_matches() -> None:
    out = WhisperLargeV3Adapter().egress(_make_output(), _make_ir(), latency_ms=150)
    assert out.provenance[-1].model_id == "openai/whisper-large-v3"


# ---------------------------------------------------------------------------
# GROUP C — Ingress correctness
# ---------------------------------------------------------------------------

def test_ingress_file_path_string_returns_audio_key() -> None:
    ir = _make_ir(content="/data/earnings_call.mp3")
    result = WhisperLargeV3Adapter().ingress(ir)
    assert result == {"audio": "/data/earnings_call.mp3"}


def test_ingress_none_content_returns_empty_audio() -> None:
    ir = _make_ir(content=None, modality="structured")
    result = WhisperLargeV3Adapter().ingress(ir)
    assert result == {"audio": ""}


def test_ingress_dict_content_passed_through_as_audio() -> None:
    path = "s3://bucket/call.wav"
    ir = _make_ir(content=path)
    result = WhisperLargeV3Adapter().ingress(ir)
    assert result["audio"] == path


def test_ingress_always_has_audio_key() -> None:
    for ir in [
        _make_ir(content="/audio/file.mp3"),
        _make_ir(content=None, modality="structured"),
        _make_ir(content="audio.flac"),
    ]:
        result = WhisperLargeV3Adapter().ingress(ir)
        assert "audio" in result


def test_ingress_returns_dict() -> None:
    result = WhisperLargeV3Adapter().ingress(_make_ir())
    assert isinstance(result, dict)


def test_ingress_file_path_matches_payload_content() -> None:
    path = "/recordings/interview.wav"
    ir = _make_ir(content=path)
    result = WhisperLargeV3Adapter().ingress(ir)
    assert result["audio"] == ir.payload.content


def test_ingress_never_returns_none() -> None:
    adapter = WhisperLargeV3Adapter()
    assert adapter.ingress(_make_ir()) is not None
    assert adapter.ingress(_make_ir(content=None, modality="structured")) is not None


# ---------------------------------------------------------------------------
# GROUP D — Edge cases (defensive egress)
# ---------------------------------------------------------------------------

def test_egress_none_model_output_no_crash() -> None:
    out = WhisperLargeV3Adapter().egress(None, _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)
    assert out.payload.content == ""


def test_egress_empty_dict_no_crash() -> None:
    out = WhisperLargeV3Adapter().egress({}, _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)
    assert out.payload.content == ""


def test_egress_missing_text_key_no_crash() -> None:
    out = WhisperLargeV3Adapter().egress({"other_key": "value"}, _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)
    assert out.payload.content == ""


def test_egress_text_value_none_sets_empty_string() -> None:
    out = WhisperLargeV3Adapter().egress({"text": None}, _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)
    assert out.payload.content == ""


def test_egress_text_value_integer_str_applied() -> None:
    out = WhisperLargeV3Adapter().egress({"text": 42}, _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)
    assert out.payload.content == "42"


def test_egress_latency_ms_zero_produces_valid_provenance() -> None:
    out = WhisperLargeV3Adapter().egress(_make_output(), _make_ir(), latency_ms=0)
    assert out.provenance[-1].latency_ms == 0
    assert out.provenance[-1].confidence == 1.0


def test_egress_whitespace_only_transcription_stripped_to_empty() -> None:
    out = WhisperLargeV3Adapter().egress({"text": "   "}, _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)
    assert out.payload.content == ""


def test_egress_provenance_appended_even_on_bad_output() -> None:
    ir = _make_ir()
    out = WhisperLargeV3Adapter().egress(None, ir, latency_ms=0)
    assert len(out.provenance) == len(ir.provenance) + 1
    assert out.provenance[-1].confidence == 1.0


def test_egress_list_output_no_crash() -> None:
    out = WhisperLargeV3Adapter().egress([{"text": "hello"}], _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)
    assert out.payload.content == ""


def test_egress_string_output_no_crash() -> None:
    out = WhisperLargeV3Adapter().egress("bare string", _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)
    assert out.payload.content == ""


# ---------------------------------------------------------------------------
# GROUP E — AdapterValidator direct pass (non-parametrized)
# ---------------------------------------------------------------------------

def test_validator_passes() -> None:
    AdapterValidator(WhisperLargeV3Adapter()).assert_valid()
