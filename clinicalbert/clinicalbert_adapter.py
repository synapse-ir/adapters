"""
SYNAPSE Adapter for medicalai/ClinicalBERT
==========================================
Model:     medicalai/ClinicalBERT
Task:      Fill-mask / masked language modeling
Domain:    clinical and medical text
Spec:      https://github.com/synapse-ir/spec
"""

from __future__ import annotations

from typing import Any

from synapse_sdk.base import AdapterBase
from synapse_sdk.types import CanonicalIR, Classification


class ClinicalBertAdapter(AdapterBase):
    """
    Adapter for medicalai/ClinicalBERT.

    The caller owns the transformers fill-mask pipeline and passes
    ``ingress(ir)["text"]`` to it. This adapter only converts between
    CanonicalIR and the pipeline's list of fill-mask candidates.
    """

    MODEL_ID = "medicalai/ClinicalBERT"
    ADAPTER_VERSION = "1.0.0"

    def ingress(self, ir: CanonicalIR) -> dict[str, Any]:
        """Return payload content as pipeline text, preserving [MASK] tokens."""
        return {"text": ir.payload.content or ""}

    def egress(
        self,
        model_output: Any,
        original_ir: CanonicalIR,
        latency_ms: int,
    ) -> CanonicalIR:
        """
        Convert Hugging Face fill-mask candidates into classifications.

        Expected model output:
            [{"score": float, "token": int, "token_str": str, "sequence": str}]

        Malformed candidates are skipped. Non-list outputs are treated as no
        candidates.
        """
        raw_candidates = model_output if isinstance(model_output, list) else []
        labels: list[Classification] = []

        for item in raw_candidates:
            if not isinstance(item, dict):
                continue

            token_str = item.get("token_str")
            if token_str is None:
                continue

            try:
                score = float(item["score"])
            except (KeyError, TypeError, ValueError):
                continue

            if score < 0.0 or score > 1.0:
                continue

            labels.append(Classification(label=str(token_str), score=score))

        confidence = labels[0].score if labels else 0.0

        updated = original_ir.clone()
        updated.payload.labels = labels
        updated.provenance.append(
            self.build_provenance(confidence=confidence, latency_ms=latency_ms)
        )

        return updated
