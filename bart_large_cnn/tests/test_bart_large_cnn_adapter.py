"""Tests for BartLargeCNNAdapter — unit behaviour and full AdapterValidator suite."""

from __future__ import annotations

import time
import uuid
from typing import Any

from bart_large_cnn_adapter import BartLargeCNNAdapter
from synapse_sdk.testing.fixtures import (
    ALL_FIXTURES,
    LEGAL_EXTRACT_BASIC,
    MAX_PROVENANCE_CHAIN,
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
    content: str | None = "The company reported record profits for the fiscal year.",
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
            task_type=TaskType.summarize,
            domain=Domain.general,
            priority=2,
            latency_budget_ms=2000,
        ),
        payload=Payload(**payload_kwargs),
        compliance_envelope=compliance or ComplianceEnvelope(),
    )


# ---------------------------------------------------------------------------
# Mock pipeline output factory
# ---------------------------------------------------------------------------

def _make_output(summary: str = "This is a test summary.") -> list[dict[str, Any]]:
    return [{"summary_text": summary}]


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_model_id() -> None:
    assert BartLargeCNNAdapter().MODEL_ID == "facebook/bart-large-cnn"


def test_adapter_version_semver() -> None:
    ver = BartLargeCNNAdapter().ADAPTER_VERSION
    parts = ver.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_default_max_length_constant() -> None:
    assert BartLargeCNNAdapter.DEFAULT_MAX_LENGTH == 130


def test_default_min_length_constant() -> None:
    assert BartLargeCNNAdapter.DEFAULT_MIN_LENGTH == 30


# ---------------------------------------------------------------------------
# ingress — standard cases
# ---------------------------------------------------------------------------

def test_ingress_returns_dict() -> None:
    result = BartLargeCNNAdapter().ingress(_make_ir())
    assert isinstance(result, dict)


def test_ingress_has_required_keys() -> None:
    result = BartLargeCNNAdapter().ingress(_make_ir())
    assert set(result.keys()) == {"text", "max_length", "min_length", "do_sample"}


def test_ingress_text_matches_payload_content() -> None:
    ir = _make_ir(content="Breaking news: scientists discover new species.")
    result = BartLargeCNNAdapter().ingress(ir)
    assert result["text"] == "Breaking news: scientists discover new species."


def test_ingress_max_length_is_130() -> None:
    result = BartLargeCNNAdapter().ingress(_make_ir())
    assert result["max_length"] == 130


def test_ingress_min_length_is_30() -> None:
    result = BartLargeCNNAdapter().ingress(_make_ir())
    assert result["min_length"] == 30


def test_ingress_do_sample_is_false() -> None:
    result = BartLargeCNNAdapter().ingress(_make_ir())
    assert result["do_sample"] is False


def test_ingress_none_content_returns_empty_text() -> None:
    ir = _make_ir(content=None, modality="structured")
    result = BartLargeCNNAdapter().ingress(ir)
    assert result["text"] == ""


def test_ingress_empty_string_content_returns_empty_text() -> None:
    ir = _make_ir(content="")
    result = BartLargeCNNAdapter().ingress(ir)
    assert result["text"] == ""


def test_ingress_never_returns_none() -> None:
    adapter = BartLargeCNNAdapter()
    assert adapter.ingress(_make_ir()) is not None
    assert adapter.ingress(_make_ir(content=None, modality="structured")) is not None


def test_ingress_multiline_text_preserved() -> None:
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    ir = _make_ir(content=text)
    result = BartLargeCNNAdapter().ingress(ir)
    assert result["text"] == text


def test_ingress_long_text_passed_through_unmodified() -> None:
    long_text = "The Senate voted on the bill. " * 200
    ir = _make_ir(content=long_text)
    result = BartLargeCNNAdapter().ingress(ir)
    assert result["text"] == long_text


# ---------------------------------------------------------------------------
# egress — return type and payload
# ---------------------------------------------------------------------------

def test_egress_returns_canonical_ir() -> None:
    out = BartLargeCNNAdapter().egress(_make_output(), _make_ir(), latency_ms=250)
    assert isinstance(out, CanonicalIR)


def test_egress_sets_payload_content_to_summary() -> None:
    out = BartLargeCNNAdapter().egress(
        _make_output("Record profits announced by company."),
        _make_ir(),
        latency_ms=250,
    )
    assert out.payload.content == "Record profits announced by company."


def test_egress_sets_content_length_to_len_of_summary() -> None:
    summary = "Record profits announced by company."
    out = BartLargeCNNAdapter().egress(_make_output(summary), _make_ir(), latency_ms=250)
    assert out.payload.content_length == len(summary)


def test_egress_payload_modality_is_text() -> None:
    out = BartLargeCNNAdapter().egress(_make_output(), _make_ir(), latency_ms=250)
    assert out.payload.modality == "text"


def test_egress_payload_modality_remains_text_for_non_text_input() -> None:
    ir = _make_ir(modality="structured")
    out = BartLargeCNNAdapter().egress(_make_output(), ir, latency_ms=250)
    assert out.payload.modality == "text"


def test_egress_original_content_replaced_not_appended() -> None:
    original_content = "The company reported record profits for the fiscal year."
    summary = "Company profits hit record highs."
    ir = _make_ir(content=original_content)
    out = BartLargeCNNAdapter().egress(_make_output(summary), ir, latency_ms=250)
    assert out.payload.content == summary
    assert original_content not in out.payload.content  # type: ignore[operator]


def test_egress_does_not_set_entities() -> None:
    out = BartLargeCNNAdapter().egress(_make_output(), _make_ir(), latency_ms=250)
    assert out.payload.entities is None


def test_egress_does_not_modify_original_ir_payload() -> None:
    original_content = "Original article text."
    ir = _make_ir(content=original_content)
    BartLargeCNNAdapter().egress(_make_output(), ir, latency_ms=250)
    assert ir.payload.content == original_content


# ---------------------------------------------------------------------------
# egress — provenance
# ---------------------------------------------------------------------------

def test_egress_appends_exactly_one_provenance_entry() -> None:
    ir = _make_ir()
    out = BartLargeCNNAdapter().egress(_make_output(), ir, latency_ms=250)
    assert len(out.provenance) == len(ir.provenance) + 1


def test_egress_provenance_confidence_is_1_0() -> None:
    out = BartLargeCNNAdapter().egress(_make_output(), _make_ir(), latency_ms=250)
    assert out.provenance[-1].confidence == 1.0


def test_egress_provenance_model_id_matches() -> None:
    out = BartLargeCNNAdapter().egress(_make_output(), _make_ir(), latency_ms=250)
    assert out.provenance[-1].model_id == "facebook/bart-large-cnn"


def test_egress_existing_provenance_not_modified() -> None:
    prior = ProvenanceEntry(
        model_id="some/prior-model",
        adapter_version="2.3.1",
        confidence=0.82,
        latency_ms=400,
        timestamp_unix=int(time.time()),
    )
    ir = _make_ir()
    ir.provenance.append(prior)

    out = BartLargeCNNAdapter().egress(_make_output(), ir, latency_ms=250)

    assert out.provenance[0].model_id == "some/prior-model"
    assert out.provenance[0].confidence == 0.82
    assert len(out.provenance) == 2
    assert out.provenance[1].model_id == "facebook/bart-large-cnn"


def test_egress_provenance_immutable_object_unchanged() -> None:
    prior = ProvenanceEntry(
        model_id="another/upstream-model",
        adapter_version="1.0.0",
        confidence=0.95,
        latency_ms=180,
        timestamp_unix=int(time.time()),
    )
    ir = _make_ir()
    ir.provenance.append(prior)
    snapshot_model_id = prior.model_id
    snapshot_confidence = prior.confidence

    BartLargeCNNAdapter().egress(_make_output(), ir, latency_ms=250)

    assert prior.model_id == snapshot_model_id
    assert prior.confidence == snapshot_confidence


def test_egress_provenance_appended_even_on_empty_output() -> None:
    ir = _make_ir()
    out = BartLargeCNNAdapter().egress([], ir, latency_ms=0)
    assert len(out.provenance) == len(ir.provenance) + 1
    assert out.provenance[-1].confidence == 1.0


# ---------------------------------------------------------------------------
# egress — task_header and compliance carried forward
# ---------------------------------------------------------------------------

def test_egress_task_header_carried_forward_unchanged() -> None:
    ir = _make_ir()
    out = BartLargeCNNAdapter().egress(_make_output(), ir, latency_ms=250)
    assert out.task_header.model_dump() == ir.task_header.model_dump()


def test_egress_compliance_envelope_carried_forward_unchanged() -> None:
    compliance = ComplianceEnvelope(
        required_tags=["gdpr", "sox"],
        pii_present=True,
        retention_policy="7y",
        data_residency=["eu-west-1"],
        purpose_limitation="financial-reporting",
    )
    ir = _make_ir(compliance=compliance)
    out = BartLargeCNNAdapter().egress(_make_output(), ir, latency_ms=250)
    assert out.compliance_envelope.model_dump() == compliance.model_dump()


def test_egress_empty_compliance_envelope_carried_forward() -> None:
    ir = _make_ir(compliance=ComplianceEnvelope())
    out = BartLargeCNNAdapter().egress(_make_output(), ir, latency_ms=250)
    assert out.compliance_envelope.model_dump() == ComplianceEnvelope().model_dump()


# ---------------------------------------------------------------------------
# egress — edge cases (graceful degradation, no raise)
# ---------------------------------------------------------------------------

def test_egress_empty_list_output_sets_content_to_empty_string() -> None:
    out = BartLargeCNNAdapter().egress([], _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)
    assert out.payload.content == ""


def test_egress_empty_list_output_sets_content_length_to_zero() -> None:
    out = BartLargeCNNAdapter().egress([], _make_ir(), latency_ms=0)
    assert out.payload.content_length == 0


def test_egress_missing_summary_text_key_sets_content_to_empty_string() -> None:
    bad_output = [{"other_key": "some value"}]
    out = BartLargeCNNAdapter().egress(bad_output, _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)
    assert out.payload.content == ""


def test_egress_missing_summary_text_key_sets_content_length_to_zero() -> None:
    bad_output = [{"other_key": "some value"}]
    out = BartLargeCNNAdapter().egress(bad_output, _make_ir(), latency_ms=0)
    assert out.payload.content_length == 0


def test_egress_none_output_does_not_raise() -> None:
    out = BartLargeCNNAdapter().egress(None, _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)
    assert out.payload.content == ""


def test_egress_dict_output_does_not_raise() -> None:
    out = BartLargeCNNAdapter().egress({"bad": "format"}, _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)
    assert out.payload.content == ""


def test_egress_string_output_does_not_raise() -> None:
    out = BartLargeCNNAdapter().egress("bare string", _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)
    assert out.payload.content == ""


def test_egress_empty_summary_text_sets_content_to_empty_string() -> None:
    out = BartLargeCNNAdapter().egress([{"summary_text": ""}], _make_ir(), latency_ms=0)
    assert out.payload.content == ""
    assert out.payload.content_length == 0


# ---------------------------------------------------------------------------
# AdapterValidator — specific fixtures
# ---------------------------------------------------------------------------

def test_validator_legal_extract_basic() -> None:
    AdapterValidator(BartLargeCNNAdapter(), fixtures=[LEGAL_EXTRACT_BASIC]).assert_valid()


def test_validator_max_provenance_chain() -> None:
    AdapterValidator(BartLargeCNNAdapter(), fixtures=[MAX_PROVENANCE_CHAIN]).assert_valid()


# ---------------------------------------------------------------------------
# AdapterValidator — all 20 standard fixtures
# ---------------------------------------------------------------------------

def test_validator_all_fixtures() -> None:
    """BartLargeCNNAdapter must pass all 13 MUST rules across all 20 standard fixtures."""
    AdapterValidator(BartLargeCNNAdapter(), fixtures=ALL_FIXTURES).assert_valid()


def test_validator_result_has_no_errors() -> None:
    result = AdapterValidator(BartLargeCNNAdapter(), fixtures=ALL_FIXTURES).run()
    assert result.passed, result.summary()
