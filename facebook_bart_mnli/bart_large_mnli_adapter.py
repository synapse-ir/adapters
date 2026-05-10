"""
SYNAPSE Adapter for facebook/bart-large-mnli
============================================
Model:     facebook/bart-large-mnli
Task:      Zero-shot text classification (classify)
Domain:    general
License:   MIT
Install:   pip install transformers torch
Spec:      https://github.com/synapse-ir/spec

Zero-shot classification via Natural Language Inference (NLI)
-------------------------------------------------------------
This model frames every classification as an NLI entailment task: for each
candidate label it checks whether the input text *entails* "This example is
about <label>." The softmax over entailment logits produces a probability
distribution across any set of labels the caller supplies at inference time —
no fine-tuning or fixed label vocabulary is required.

Candidate labels
----------------
Labels are read from ``ir.task_header.candidate_labels`` at inference time.
Any list of strings is valid: domain names, intent categories, topic tags, etc.
When no labels are provided the adapter defaults to
``["positive", "negative", "neutral"]``.

Full ranked output
------------------
Unlike single-label classifiers, the pipeline returns ALL candidate labels
sorted by descending probability. ``payload.labels`` is populated with a
:class:`~synapse_sdk.types.Classification` entry for every label so callers
can inspect the full probability distribution, not just the top prediction.

MIT license
-----------
The model weights are released under the MIT license. See
https://huggingface.co/facebook/bart-large-mnli for the model card.
"""

from __future__ import annotations

from typing import Any

from synapse_sdk import AdapterBase, CanonicalIR
from synapse_sdk.types import Classification

_DEFAULT_LABELS: list[str] = ["positive", "negative", "neutral"]


class BartLargeMnliAdapter(AdapterBase):
    """Adapter for facebook/bart-large-mnli zero-shot classification."""

    MODEL_ID: str = "facebook/bart-large-mnli"
    ADAPTER_VERSION: str = "1.0.0"

    # -------------------------------------------------------------------------
    # ingress
    # -------------------------------------------------------------------------

    def ingress(self, ir: CanonicalIR) -> dict[str, Any]:
        """
        Prepare the pipeline input dict from the canonical IR.

        Returns::

            {
                "text": str,                  # payload.content; "" when None
                "candidate_labels": list[str] # from task_header or default
            }
        """
        candidate_labels: list[str] = (
            ir.task_header.candidate_labels or _DEFAULT_LABELS
        )
        return {
            "text": ir.payload.content or "",
            "candidate_labels": candidate_labels,
        }

    # -------------------------------------------------------------------------
    # egress
    # -------------------------------------------------------------------------

    def egress(
        self,
        model_output: Any,
        original_ir: CanonicalIR,
        latency_ms: int,
    ) -> CanonicalIR:
        """
        Convert the zero-shot pipeline output dict into a CanonicalIR.

        Expected ``model_output`` schema::

            {
                "sequence": str,
                "labels":   list[str],   # sorted by descending score
                "scores":   list[float], # same order as labels, sums to ~1.0
            }

        All labels are stored in ``payload.labels`` as
        :class:`~synapse_sdk.types.Classification` objects.
        Confidence is ``scores[0]`` — the probability of the top-ranked label.

        If ``model_output`` is ``None``, not a dict, or missing required keys,
        ``payload.labels`` is set to ``[]`` and provenance confidence is
        ``0.0`` rather than raising.
        """
        classifications: list[Classification] = []
        confidence: float = 0.0

        try:
            if isinstance(model_output, dict):
                labels = model_output.get("labels", [])
                scores = model_output.get("scores", [])
                classifications = [
                    Classification(label=str(lbl), score=float(sc))
                    for lbl, sc in zip(labels, scores)
                ]
                confidence = float(scores[0]) if scores else 0.0
        except Exception:  # noqa: BLE001
            pass

        updated = original_ir.clone()
        updated.payload.labels = classifications
        updated.provenance.append(
            self.build_provenance(confidence=confidence, latency_ms=latency_ms)
        )
        return updated
