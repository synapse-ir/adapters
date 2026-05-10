"""
SYNAPSE Adapter for ProsusAI/finbert
=====================================
Model:     ProsusAI/finbert
Task:      Financial sentiment classification (text-classification)
Domain:    finance
License:   Apache 2.0
Install:   pip install transformers torch
Spec:      https://github.com/synapse-ir/spec
"""

from __future__ import annotations

from typing import Any

from synapse_sdk import AdapterBase, CanonicalIR
from synapse_sdk.types import Classification


class FinbertAdapter(AdapterBase):
    """
    Adapter for ProsusAI/finbert.

    FinBERT is a BERT model fine-tuned on financial news and earnings call
    transcripts for three-class sentiment classification. It determines whether
    a piece of financial text expresses positive, negative, or neutral sentiment
    toward the subject (a company, sector, or market event).

    Input / output format
    ---------------------
    ingress receives a CanonicalIR with ``payload.modality="text"`` and
    returns ``{"text": str}`` for the transformers pipeline.

    egress receives ``[{"label": str, "score": float}]`` — the top-1 list
    produced by the pipeline — and returns a CanonicalIR with
    ``payload.labels`` populated by a single :class:`Classification` object
    whose ``label`` is one of ``"positive"``, ``"negative"``, ``"neutral"``
    and whose ``score`` is the softmax probability of that class.

    Domain: finance
    ---------------
    Labels encode the market sentiment expressed in the input text:

    - ``"positive"`` — the text conveys optimism, growth, or beat expectations.
    - ``"negative"`` — the text conveys pessimism, loss, or missed expectations.
    - ``"neutral"``  — the text is factual or non-directional.

    Architecture — pure functions, caller-owned pipeline
    ----------------------------------------------------
    The adapter never loads or invokes the model. The caller owns the
    transformers ``pipeline`` instance and drives inference. The adapter
    provides two pure transformation steps:

      ingress  — converts CanonicalIR to the dict the caller passes to the
                 pipeline
      egress   — converts the pipeline list result back into a CanonicalIR

    Typical caller pattern::

        from transformers import pipeline
        from finbert_adapter import FinbertAdapter

        pipe    = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        adapter = FinbertAdapter()

        model_input  = adapter.ingress(ir)
        model_output = pipe(model_input["text"])
        result_ir    = adapter.egress(model_output, ir, latency_ms=latency_ms)

        label = result_ir.payload.labels[0].label   # "positive" | "negative" | "neutral"
        score = result_ir.payload.labels[0].score   # float in [0.0, 1.0]

    PII
    ---
    FinBERT classifies financial sentiment and does not extract person entities.
    ``compliance_envelope.pii_present`` is never upgraded to ``True`` by this
    adapter.

    Confidence
    ----------
    Confidence equals the softmax score of the top predicted class, matching
    the single element returned by the pipeline.  A confidence of ``0.0`` is
    recorded when model output is malformed or empty — not treated as an error.
    """

    MODEL_ID: str = "ProsusAI/finbert"
    ADAPTER_VERSION: str = "1.0.0"

    # -------------------------------------------------------------------------
    # ingress
    # -------------------------------------------------------------------------

    def ingress(self, ir: CanonicalIR) -> dict[str, Any]:
        """
        Prepare the pipeline input dict from the canonical IR.

        Returns::

            {"text": str}   # payload.content; "" when content is None

        The caller passes ``result["text"]`` to the sentiment-analysis pipeline.
        """
        return {"text": ir.payload.content or ""}

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
        Convert ``[{"label": str, "score": float}]`` from the pipeline into a
        CanonicalIR with ``payload.labels`` set to a single
        :class:`Classification` object.

        If ``model_output`` is empty, ``None``, or otherwise malformed,
        ``payload.labels`` is set to ``[]`` and provenance confidence is
        ``0.0`` rather than raising.
        """
        labels: list[Classification] = []
        confidence: float = 0.0

        try:
            if isinstance(model_output, list) and len(model_output) > 0:
                item = model_output[0]
                if isinstance(item, dict):
                    label = str(item.get("label", ""))
                    score = float(item.get("score", 0.0))
                    labels = [Classification(label=label, score=score)]
                    confidence = score
        except Exception:  # noqa: BLE001
            pass

        updated = original_ir.clone()
        updated.payload.labels = labels

        updated.provenance.append(
            self.build_provenance(confidence=confidence, latency_ms=latency_ms)
        )

        return updated
