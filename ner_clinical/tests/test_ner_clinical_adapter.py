"""Tests for NerClinicalAdapter — unit behaviour and full AdapterValidator suite."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from ner_clinical_adapter import NerClinicalAdapter
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
    TaskHeader,
    TaskType,
)
from synapse_sdk.validator import AdapterValidator


# ---------------------------------------------------------------------------
# MockAnnotation — simulates a Spark NLP Annotation object from NerConverter
# ---------------------------------------------------------------------------

class MockAnnotation:
    def __init__(self, result: str, begin: int, end: int, entity: str, confidence: float):
        self.result = result
        self.begin = begin
        self.end = end
        self.metadata = {"entity": entity, "confidence": str(confidence)}


# ---------------------------------------------------------------------------
# IR factory helpers
# ---------------------------------------------------------------------------

def _make_ir(
    content: str | None = "The patient has diabetes and was given metformin.",
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
            task_type=TaskType.extract,
            domain=Domain.medical,
            priority=2,
            latency_budget_ms=1000,
        ),
        payload=Payload(**payload_kwargs),
        compliance_envelope=compliance or ComplianceEnvelope(),
    )


# ---------------------------------------------------------------------------
# Canned annotation sets
# ---------------------------------------------------------------------------

_PROBLEM_ANN = MockAnnotation("diabetes", 15, 22, "PROBLEM", 0.9876)
_TEST_ANN    = MockAnnotation("blood glucose", 50, 62, "TEST", 0.9543)
_TREAT_ANN   = MockAnnotation("metformin", 80, 88, "TREATMENT", 0.9712)

_MULTI_ANNS = [_PROBLEM_ANN, _TEST_ANN, _TREAT_ANN]


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_model_id():
    assert NerClinicalAdapter().MODEL_ID == "johnsnowlabs/ner_clinical"


def test_adapter_version_semver():
    ver = NerClinicalAdapter().ADAPTER_VERSION
    parts = ver.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


# ---------------------------------------------------------------------------
# ingress
# ---------------------------------------------------------------------------

def test_ingress_returns_dict():
    result = NerClinicalAdapter().ingress(_make_ir())
    assert isinstance(result, dict)


def test_ingress_standard_text():
    ir = _make_ir(content="Patient presents with chest pain.")
    result = NerClinicalAdapter().ingress(ir)
    assert result == {"text": "Patient presents with chest pain."}


def test_ingress_none_content_returns_empty_string():
    ir = _make_ir(modality="structured")
    result = NerClinicalAdapter().ingress(ir)
    assert result == {"text": ""}


def test_ingress_empty_string_content():
    ir = _make_ir(content="")
    result = NerClinicalAdapter().ingress(ir)
    assert result == {"text": ""}


def test_ingress_never_returns_none():
    adapter = NerClinicalAdapter()
    assert adapter.ingress(_make_ir()) is not None
    assert adapter.ingress(_make_ir(content=None, modality="structured")) is not None


# ---------------------------------------------------------------------------
# egress — return type and structure
# ---------------------------------------------------------------------------

def test_egress_returns_canonical_ir():
    out = NerClinicalAdapter().egress([_PROBLEM_ANN], _make_ir(), latency_ms=100)
    assert isinstance(out, CanonicalIR)


def test_egress_appends_exactly_one_provenance():
    ir = _make_ir()
    out = NerClinicalAdapter().egress([_PROBLEM_ANN], ir, latency_ms=100)
    assert len(out.provenance) == len(ir.provenance) + 1


def test_egress_provenance_model_id():
    out = NerClinicalAdapter().egress([_PROBLEM_ANN], _make_ir(), latency_ms=50)
    assert out.provenance[-1].model_id == "johnsnowlabs/ner_clinical"


def test_egress_carries_task_header():
    ir = _make_ir()
    out = NerClinicalAdapter().egress([_PROBLEM_ANN], ir, latency_ms=50)
    assert out.task_header.model_dump() == ir.task_header.model_dump()


# ---------------------------------------------------------------------------
# egress — entity mapping
# ---------------------------------------------------------------------------

def test_egress_single_problem_entity():
    out = NerClinicalAdapter().egress([_PROBLEM_ANN], _make_ir(), latency_ms=50)
    assert out.payload.entities is not None
    assert len(out.payload.entities) == 1
    e = out.payload.entities[0]
    assert e.text == "diabetes"
    assert e.label == "PROBLEM"
    assert e.start == 15
    assert e.end == 22
    assert pytest.approx(e.confidence, abs=1e-4) == 0.9876


def test_egress_single_test_entity():
    out = NerClinicalAdapter().egress([_TEST_ANN], _make_ir(), latency_ms=50)
    assert len(out.payload.entities) == 1
    e = out.payload.entities[0]
    assert e.text == "blood glucose"
    assert e.label == "TEST"
    assert e.start == 50
    assert e.end == 62


def test_egress_single_treatment_entity():
    out = NerClinicalAdapter().egress([_TREAT_ANN], _make_ir(), latency_ms=50)
    assert len(out.payload.entities) == 1
    e = out.payload.entities[0]
    assert e.text == "metformin"
    assert e.label == "TREATMENT"
    assert e.start == 80
    assert e.end == 88


def test_egress_multiple_entities_different_types():
    out = NerClinicalAdapter().egress(_MULTI_ANNS, _make_ir(), latency_ms=50)
    assert len(out.payload.entities) == 3
    labels = {e.label for e in out.payload.entities}
    assert labels == {"PROBLEM", "TEST", "TREATMENT"}


def test_egress_empty_model_output_no_entities():
    out = NerClinicalAdapter().egress([], _make_ir(), latency_ms=50)
    # entities should be None (no results) — adapter must not raise
    assert out.payload.entities is None or out.payload.entities == []


def test_egress_empty_model_output_does_not_raise():
    ir = _make_ir()
    out = NerClinicalAdapter().egress([], ir, latency_ms=50)
    assert isinstance(out, CanonicalIR)


def test_egress_non_list_model_output_treated_as_empty():
    # The validator passes a dict as dummy output — must not raise
    dummy = {"result": "test", "model_conf": 0.9}
    out = NerClinicalAdapter().egress(dummy, _make_ir(), latency_ms=42)
    assert isinstance(out, CanonicalIR)
    assert out.payload.entities is None or out.payload.entities == []


# ---------------------------------------------------------------------------
# egress — confidence handling
# ---------------------------------------------------------------------------

def test_egress_confidence_string_cast_to_float():
    ann = MockAnnotation("hypertension", 0, 11, "PROBLEM", 0.9500)
    out = NerClinicalAdapter().egress([ann], _make_ir(), latency_ms=50)
    assert isinstance(out.payload.entities[0].confidence, float)
    assert pytest.approx(out.payload.entities[0].confidence, abs=1e-4) == 0.9500


def test_egress_malformed_confidence_defaults_to_zero():
    ann = MockAnnotation("hypertension", 0, 11, "PROBLEM", 0.9)
    ann.metadata["confidence"] = "not_a_float"
    out = NerClinicalAdapter().egress([ann], _make_ir(), latency_ms=50)
    assert out.payload.entities[0].confidence == 0.0


def test_egress_confidence_above_one_clamped():
    ann = MockAnnotation("diabetes", 0, 7, "PROBLEM", 0.9)
    ann.metadata["confidence"] = "1.5"
    out = NerClinicalAdapter().egress([ann], _make_ir(), latency_ms=50)
    assert out.payload.entities[0].confidence == 1.0


def test_egress_confidence_below_zero_clamped():
    ann = MockAnnotation("diabetes", 0, 7, "PROBLEM", 0.9)
    ann.metadata["confidence"] = "-0.2"
    out = NerClinicalAdapter().egress([ann], _make_ir(), latency_ms=50)
    assert out.payload.entities[0].confidence == 0.0


def test_egress_unknown_label_silently_skipped():
    unknown = MockAnnotation("some text", 0, 9, "UNKNOWN_LABEL", 0.9)
    out = NerClinicalAdapter().egress([unknown, _TEST_ANN], _make_ir(), latency_ms=50)
    assert len(out.payload.entities) == 1
    assert out.payload.entities[0].label == "TEST"


# ---------------------------------------------------------------------------
# egress — confidence aggregation for provenance
# ---------------------------------------------------------------------------

def test_egress_empty_output_provenance_confidence_zero():
    out = NerClinicalAdapter().egress([], _make_ir(), latency_ms=50)
    assert out.provenance[-1].confidence == 0.0


def test_egress_single_entity_confidence_is_entity_confidence():
    out = NerClinicalAdapter().egress([_PROBLEM_ANN], _make_ir(), latency_ms=50)
    assert pytest.approx(out.provenance[-1].confidence, abs=1e-4) == 0.9876


def test_egress_multiple_entity_confidence_is_mean():
    out = NerClinicalAdapter().egress(_MULTI_ANNS, _make_ir(), latency_ms=50)
    expected = (0.9876 + 0.9543 + 0.9712) / 3
    assert pytest.approx(out.provenance[-1].confidence, abs=1e-4) == expected


# ---------------------------------------------------------------------------
# egress — provenance immutability
# ---------------------------------------------------------------------------

def test_egress_existing_provenance_not_modified():
    from synapse_sdk.types import ProvenanceEntry
    import time

    prior = ProvenanceEntry(
        model_id="some/prior-model",
        adapter_version="1.0.0",
        confidence=0.80,
        latency_ms=200,
        timestamp_unix=int(time.time()),
    )
    ir = _make_ir()
    ir.provenance.append(prior)

    out = NerClinicalAdapter().egress([_PROBLEM_ANN], ir, latency_ms=50)

    # Original entry untouched
    assert out.provenance[0].model_id == "some/prior-model"
    assert out.provenance[0].confidence == 0.80
    # New entry appended
    assert len(out.provenance) == 2
    assert out.provenance[1].model_id == "johnsnowlabs/ner_clinical"


# ---------------------------------------------------------------------------
# egress — task_header and compliance carried forward
# ---------------------------------------------------------------------------

def test_egress_task_header_carried_forward_unchanged():
    ir = _make_ir()
    out = NerClinicalAdapter().egress([_TEST_ANN], ir, latency_ms=50)
    assert out.task_header.model_dump() == ir.task_header.model_dump()


def test_egress_compliance_carried_unchanged_when_no_pii():
    compliance = ComplianceEnvelope(
        required_tags=["hipaa"],
        pii_present=False,
        retention_policy="7y",
        data_residency=["us-east-1"],
        purpose_limitation="treatment",
    )
    ir = _make_ir(compliance=compliance)
    # TEST and TREATMENT — no PII entities
    out = NerClinicalAdapter().egress([_TEST_ANN, _TREAT_ANN], ir, latency_ms=50)
    assert out.compliance_envelope.model_dump() == compliance.model_dump()


# ---------------------------------------------------------------------------
# egress — G-S04 PII handling
# ---------------------------------------------------------------------------

def test_egress_pii_present_propagated_when_incoming_true():
    ir = _make_ir(compliance=ComplianceEnvelope(pii_present=True))
    out = NerClinicalAdapter().egress([], ir, latency_ms=50)
    assert out.compliance_envelope.pii_present is True


def test_egress_pii_present_set_when_problem_entity_extracted():
    ir = _make_ir(compliance=ComplianceEnvelope(pii_present=None))
    out = NerClinicalAdapter().egress([_PROBLEM_ANN], ir, latency_ms=50)
    assert out.compliance_envelope.pii_present is True


def test_egress_pii_not_set_for_test_entity_only():
    ir = _make_ir(compliance=ComplianceEnvelope(pii_present=None))
    out = NerClinicalAdapter().egress([_TEST_ANN], ir, latency_ms=50)
    assert out.compliance_envelope.pii_present is not True


def test_egress_pii_not_set_for_treatment_entity_only():
    ir = _make_ir(compliance=ComplianceEnvelope(pii_present=None))
    out = NerClinicalAdapter().egress([_TREAT_ANN], ir, latency_ms=50)
    assert out.compliance_envelope.pii_present is not True


def test_egress_pii_upgrade_preserves_other_compliance_fields():
    compliance = ComplianceEnvelope(
        pii_present=None,
        required_tags=["hipaa"],
        retention_policy="7y",
        data_residency=["us-east-1"],
        purpose_limitation="treatment",
    )
    ir = _make_ir(compliance=compliance)
    out = NerClinicalAdapter().egress([_PROBLEM_ANN], ir, latency_ms=50)
    env = out.compliance_envelope
    assert env.pii_present is True
    assert env.required_tags == ["hipaa"]
    assert env.retention_policy == "7y"
    assert env.data_residency == ["us-east-1"]
    assert env.purpose_limitation == "treatment"


def test_egress_pii_already_true_stays_true_with_problem_entity():
    ir = _make_ir(compliance=ComplianceEnvelope(pii_present=True))
    out = NerClinicalAdapter().egress([_PROBLEM_ANN], ir, latency_ms=50)
    assert out.compliance_envelope.pii_present is True


# ---------------------------------------------------------------------------
# egress — content preserved
# ---------------------------------------------------------------------------

def test_egress_payload_content_not_overwritten():
    ir = _make_ir(content="Original clinical note.")
    out = NerClinicalAdapter().egress([_PROBLEM_ANN], ir, latency_ms=50)
    assert out.payload.content == "Original clinical note."


# ---------------------------------------------------------------------------
# AdapterValidator — specific fixtures
# ---------------------------------------------------------------------------

def test_validator_legal_extract_basic():
    AdapterValidator(NerClinicalAdapter(), fixtures=[LEGAL_EXTRACT_BASIC]).assert_valid()


def test_validator_general_embed_large():
    AdapterValidator(NerClinicalAdapter(), fixtures=[GENERAL_EMBED_LARGE]).assert_valid()


# ---------------------------------------------------------------------------
# AdapterValidator — all 20 standard fixtures (§9 G-S06)
# ---------------------------------------------------------------------------

def test_validator_all_fixtures():
    """NerClinicalAdapter must pass all 13 MUST rules across all 20 standard fixtures."""
    AdapterValidator(NerClinicalAdapter(), fixtures=ALL_FIXTURES).assert_valid()


def test_validator_result_has_no_errors():
    result = AdapterValidator(NerClinicalAdapter(), fixtures=ALL_FIXTURES).run()
    assert result.passed, result.summary()
