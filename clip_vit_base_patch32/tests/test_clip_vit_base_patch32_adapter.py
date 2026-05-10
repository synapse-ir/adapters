"""Tests for ClipVitBasePatch32Adapter — unit behaviour and full AdapterValidator suite."""

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
from clip_vit_base_patch32_adapter import ClipVitBasePatch32Adapter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_LABELS = ["object", "animal", "vehicle", "person", "food"]

_MOCK_OUTPUT = [
    {"score": 0.9993, "label": "a photo of a cat"},
    {"score": 0.0003, "label": "a photo of a dog"},
    {"score": 0.0002, "label": "a photo of a car"},
]

# ---------------------------------------------------------------------------
# IR factory
# ---------------------------------------------------------------------------


def _make_ir(
    content: str | None = "http://images.cocodataset.org/val2017/000000039769.jpg",
    candidate_labels: list[str] | None = None,
    compliance: ComplianceEnvelope | None = None,
) -> CanonicalIR:
    return CanonicalIR(
        ir_version="1.0.0",
        message_id=str(uuid.uuid4()),
        task_header=TaskHeader(
            task_type=TaskType.classify,
            domain=Domain.general,
            priority=2,
            latency_budget_ms=5000,
            candidate_labels=candidate_labels,
        ),
        payload=Payload(modality="text", content=content),
        compliance_envelope=compliance or ComplianceEnvelope(),
    )


# ---------------------------------------------------------------------------
# GROUP A — Validator fixture suite (parametrized over all standard fixtures)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ALL_FIXTURES)
def test_all_fixtures(fixture: CanonicalIR) -> None:
    AdapterValidator(ClipVitBasePatch32Adapter()).assert_valid_on(fixture)


# ---------------------------------------------------------------------------
# GROUP B — Egress correctness (mock output only)
# ---------------------------------------------------------------------------


def test_egress_all_three_labels_stored_in_payload_labels() -> None:
    out = ClipVitBasePatch32Adapter().egress(_MOCK_OUTPUT, _make_ir(), latency_ms=100)
    assert out.payload.labels is not None
    assert len(out.payload.labels) == 3


def test_egress_labels_index_0_label() -> None:
    out = ClipVitBasePatch32Adapter().egress(_MOCK_OUTPUT, _make_ir(), latency_ms=100)
    assert out.payload.labels[0].label == "a photo of a cat"  # type: ignore[index]


def test_egress_labels_index_0_score() -> None:
    out = ClipVitBasePatch32Adapter().egress(_MOCK_OUTPUT, _make_ir(), latency_ms=100)
    assert out.payload.labels[0].score == 0.9993  # type: ignore[index]


def test_egress_labels_index_1_label() -> None:
    out = ClipVitBasePatch32Adapter().egress(_MOCK_OUTPUT, _make_ir(), latency_ms=100)
    assert out.payload.labels[1].label == "a photo of a dog"  # type: ignore[index]


def test_egress_labels_index_2_label() -> None:
    out = ClipVitBasePatch32Adapter().egress(_MOCK_OUTPUT, _make_ir(), latency_ms=100)
    assert out.payload.labels[2].label == "a photo of a car"  # type: ignore[index]


def test_egress_confidence_equals_top_score() -> None:
    out = ClipVitBasePatch32Adapter().egress(_MOCK_OUTPUT, _make_ir(), latency_ms=100)
    assert out.provenance[-1].confidence == 0.9993


def test_egress_provenance_length_is_original_plus_one() -> None:
    ir = _make_ir()
    out = ClipVitBasePatch32Adapter().egress(_MOCK_OUTPUT, ir, latency_ms=100)
    assert len(out.provenance) == len(ir.provenance) + 1


def test_egress_task_header_carried_forward_unchanged() -> None:
    ir = _make_ir(candidate_labels=["cat", "dog"])
    out = ClipVitBasePatch32Adapter().egress(_MOCK_OUTPUT, ir, latency_ms=100)
    assert out.task_header.model_dump() == ir.task_header.model_dump()


def test_egress_compliance_envelope_carried_forward_unchanged() -> None:
    compliance = ComplianceEnvelope(
        required_tags=["internal"],
        pii_present=False,
        retention_policy="1y",
        data_residency=["eu-west-1"],
        purpose_limitation="classification",
    )
    ir = _make_ir(compliance=compliance)
    out = ClipVitBasePatch32Adapter().egress(_MOCK_OUTPUT, ir, latency_ms=100)
    assert out.compliance_envelope.model_dump() == compliance.model_dump()


def test_egress_pii_present_remains_false() -> None:
    ir = _make_ir(compliance=ComplianceEnvelope(pii_present=False))
    out = ClipVitBasePatch32Adapter().egress(_MOCK_OUTPUT, ir, latency_ms=100)
    assert out.compliance_envelope.pii_present is False


def test_egress_payload_content_not_modified() -> None:
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    ir = _make_ir(content=url)
    out = ClipVitBasePatch32Adapter().egress(_MOCK_OUTPUT, ir, latency_ms=100)
    assert out.payload.content == url


def test_egress_latency_ms_stored_correctly_in_provenance() -> None:
    out = ClipVitBasePatch32Adapter().egress(_MOCK_OUTPUT, _make_ir(), latency_ms=777)
    assert out.provenance[-1].latency_ms == 777


def test_egress_single_label_output_stored_correctly() -> None:
    single = [{"score": 0.9999, "label": "cat"}]
    out = ClipVitBasePatch32Adapter().egress(single, _make_ir(), latency_ms=50)
    assert out.payload.labels is not None
    assert len(out.payload.labels) == 1
    assert out.payload.labels[0].label == "cat"  # type: ignore[index]
    assert out.payload.labels[0].score == 0.9999  # type: ignore[index]


def test_egress_two_label_output_stored_correctly() -> None:
    two = [
        {"score": 0.75, "label": "animal"},
        {"score": 0.25, "label": "vehicle"},
    ]
    out = ClipVitBasePatch32Adapter().egress(two, _make_ir(), latency_ms=80)
    assert out.payload.labels is not None
    assert len(out.payload.labels) == 2
    assert out.payload.labels[0].label == "animal"  # type: ignore[index]
    assert out.payload.labels[1].label == "vehicle"  # type: ignore[index]


# ---------------------------------------------------------------------------
# GROUP C — Ingress
# ---------------------------------------------------------------------------


def test_ingress_candidate_labels_present_passed_through() -> None:
    ir = _make_ir(candidate_labels=["cat", "dog", "car"])
    result = ClipVitBasePatch32Adapter().ingress(ir)
    assert result["candidate_labels"] == ["cat", "dog", "car"]


def test_ingress_candidate_labels_none_uses_default() -> None:
    ir = _make_ir(candidate_labels=None)
    result = ClipVitBasePatch32Adapter().ingress(ir)
    assert result["candidate_labels"] == _DEFAULT_LABELS


def test_ingress_candidate_labels_empty_list_uses_default() -> None:
    ir = _make_ir(candidate_labels=[])
    result = ClipVitBasePatch32Adapter().ingress(ir)
    assert result["candidate_labels"] == _DEFAULT_LABELS


def test_ingress_image_content_present_stored_under_image_key() -> None:
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    ir = _make_ir(content=url)
    result = ClipVitBasePatch32Adapter().ingress(ir)
    assert result["image"] == url


def test_ingress_image_content_none_returns_empty_string() -> None:
    ir = CanonicalIR(
        ir_version="1.0.0",
        message_id=str(uuid.uuid4()),
        task_header=TaskHeader(
            task_type=TaskType.classify,
            domain=Domain.general,
            priority=2,
            latency_budget_ms=5000,
        ),
        payload=Payload(modality="structured", data={"key": "value"}),
        compliance_envelope=ComplianceEnvelope(),
    )
    result = ClipVitBasePatch32Adapter().ingress(ir)
    assert result["image"] == ""


def test_ingress_url_string_passed_through_unchanged() -> None:
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg"
    ir = _make_ir(content=url)
    result = ClipVitBasePatch32Adapter().ingress(ir)
    assert result["image"] == url


def test_ingress_output_always_has_image_key() -> None:
    for ir in [_make_ir(content="/path/to/image.jpg"), _make_ir(content="http://example.com/img.png")]:
        result = ClipVitBasePatch32Adapter().ingress(ir)
        assert "image" in result


def test_ingress_output_always_has_candidate_labels_key() -> None:
    for ir in [_make_ir(candidate_labels=["a", "b"]), _make_ir(candidate_labels=None)]:
        result = ClipVitBasePatch32Adapter().ingress(ir)
        assert "candidate_labels" in result


def test_ingress_file_path_passed_through_unchanged() -> None:
    path = "/data/images/cat.jpg"
    ir = _make_ir(content=path)
    result = ClipVitBasePatch32Adapter().ingress(ir)
    assert result["image"] == path


# ---------------------------------------------------------------------------
# GROUP D — Edge cases
# ---------------------------------------------------------------------------


def test_egress_none_model_output_empty_labels_no_crash() -> None:
    out = ClipVitBasePatch32Adapter().egress(None, _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)
    assert out.payload.labels == []


def test_egress_none_model_output_confidence_zero() -> None:
    out = ClipVitBasePatch32Adapter().egress(None, _make_ir(), latency_ms=0)
    assert out.provenance[-1].confidence == 0.0


def test_egress_empty_list_output_empty_labels_no_crash() -> None:
    out = ClipVitBasePatch32Adapter().egress([], _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)
    assert out.payload.labels == []


def test_egress_empty_list_output_confidence_zero() -> None:
    out = ClipVitBasePatch32Adapter().egress([], _make_ir(), latency_ms=0)
    assert out.provenance[-1].confidence == 0.0


def test_egress_non_list_dict_output_no_crash() -> None:
    out = ClipVitBasePatch32Adapter().egress({"score": 0.9, "label": "cat"}, _make_ir(), latency_ms=0)
    assert isinstance(out, CanonicalIR)


def test_egress_item_missing_label_key_defaults_to_empty_string() -> None:
    output = [{"score": 0.8}]
    out = ClipVitBasePatch32Adapter().egress(output, _make_ir(), latency_ms=0)
    assert out.payload.labels is not None
    assert out.payload.labels[0].label == ""  # type: ignore[index]


def test_egress_item_missing_score_key_defaults_to_zero() -> None:
    output = [{"label": "cat"}]
    out = ClipVitBasePatch32Adapter().egress(output, _make_ir(), latency_ms=0)
    assert out.payload.labels is not None
    assert out.payload.labels[0].score == 0.0  # type: ignore[index]


def test_egress_non_dict_item_in_list_skipped_gracefully() -> None:
    output: list[Any] = [
        {"score": 0.9, "label": "cat"},
        "not_a_dict",
        {"score": 0.1, "label": "dog"},
    ]
    out = ClipVitBasePatch32Adapter().egress(output, _make_ir(), latency_ms=0)
    assert out.payload.labels is not None
    assert len(out.payload.labels) == 2


def test_egress_latency_ms_zero_valid_provenance() -> None:
    out = ClipVitBasePatch32Adapter().egress(_MOCK_OUTPUT, _make_ir(), latency_ms=0)
    assert out.provenance[-1].latency_ms == 0
    assert isinstance(out, CanonicalIR)


def test_egress_does_not_mutate_original_ir() -> None:
    ir = _make_ir(content="http://example.com/cat.jpg")
    original_content = ir.payload.content
    original_prov_len = len(ir.provenance)
    ClipVitBasePatch32Adapter().egress(_MOCK_OUTPUT, ir, latency_ms=100)
    assert ir.payload.content == original_content
    assert len(ir.provenance) == original_prov_len


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

    out = ClipVitBasePatch32Adapter().egress(_MOCK_OUTPUT, ir, latency_ms=150)

    assert out.provenance[0].model_id == "some/prior-model"
    assert out.provenance[0].confidence == 0.85
    assert len(out.provenance) == 2
    assert out.provenance[1].model_id == "openai/clip-vit-base-patch32"


def test_egress_returns_canonical_ir() -> None:
    out = ClipVitBasePatch32Adapter().egress(_MOCK_OUTPUT, _make_ir(), latency_ms=100)
    assert isinstance(out, CanonicalIR)


def test_egress_provenance_model_id_matches() -> None:
    out = ClipVitBasePatch32Adapter().egress(_MOCK_OUTPUT, _make_ir(), latency_ms=100)
    assert out.provenance[-1].model_id == "openai/clip-vit-base-patch32"


def test_egress_payload_entities_not_modified() -> None:
    ir = _make_ir()
    assert ir.payload.entities is None
    out = ClipVitBasePatch32Adapter().egress(_MOCK_OUTPUT, ir, latency_ms=100)
    assert out.payload.entities is None


def test_egress_payload_score_not_modified() -> None:
    ir = _make_ir()
    assert ir.payload.score is None
    out = ClipVitBasePatch32Adapter().egress(_MOCK_OUTPUT, ir, latency_ms=100)
    assert out.payload.score is None


# ---------------------------------------------------------------------------
# GROUP E — AdapterValidator direct pass (non-parametrized)
# ---------------------------------------------------------------------------


def test_validator_passes() -> None:
    AdapterValidator(ClipVitBasePatch32Adapter()).assert_valid()
