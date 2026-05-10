"""
SYNAPSE Adapter for cardiffnlp/twitter-roberta-base-sentiment-latest
=====================================================================
Model:     cardiffnlp/twitter-roberta-base-sentiment-latest
Task:      Social media sentiment classification (classify)
Domain:    conversational
License:   CC BY 4.0 (attribution required)
Install:   pip install transformers torch
Spec:      https://github.com/synapse-ir/spec
"""

from __future__ import annotations

from typing import Any

from synapse_sdk import AdapterBase, CanonicalIR
from synapse_sdk.types import Classification


class TwitterRobertaSentimentAdapter(AdapterBase):
    """
    Adapter for cardiffnlp/twitter-roberta-base-sentiment-latest.

    Classifies the sentiment of social-media text as Negative, Neutral, or
    Positive. The model is a RoBERTa-base checkpoint fine-tuned on 124 million
    tweets collected between January 2018 and December 2021. It is optimised
    for the informal language, hashtags, mentions, and slang common on Twitter
    (X) and other short-form social platforms.

    Social media domain
    -------------------
    Unlike news-sentiment models (e.g. FinBERT), this model is exposed to
    conversational text with abbreviations, emojis, and code-mixed content. It
    generalises well to Reddit, product reviews, and other user-generated
    content but degrades on formal or domain-specific prose (legal, medical,
    financial).

    Title-cased label format
    ------------------------
    The pipeline returns labels in title case: ``'Negative'``, ``'Neutral'``,
    ``'Positive'``. This is different from some other sentiment models that use
    lowercase (e.g. finbert). Callers that switch between adapters must account
    for this difference when matching labels programmatically.

    Input / output format
    ---------------------
    ingress receives a CanonicalIR with ``payload.modality="text"`` and
    returns ``{"text": str}`` for the transformers pipeline.

    egress receives ``[{"label": str, "score": float}]`` — the top-1 list
    produced by the pipeline — and returns a CanonicalIR with
    ``payload.labels`` populated by a single :class:`Classification` object.

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
        from twitter_roberta_sentiment_adapter import TwitterRobertaSentimentAdapter

        pipe    = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        )
        adapter = TwitterRobertaSentimentAdapter()

        model_input  = adapter.ingress(ir)
        model_output = pipe(model_input["text"])
        result_ir    = adapter.egress(model_output, ir, latency_ms=latency_ms)

        label = result_ir.payload.labels[0].label  # 'Negative' | 'Neutral' | 'Positive'
        score = result_ir.payload.labels[0].score  # float in [0.0, 1.0]

    License
    -------
    The model weights are released under CC BY 4.0. Attribution to the Cardiff
    NLP group is required when redistributing or building products on top of
    this model. See https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

    PII
    ---
    Sentiment classification does not extract person entities.
    ``compliance_envelope.pii_present`` is never upgraded to ``True`` by this
    adapter.

    Confidence
    ----------
    Confidence equals the softmax score of the top predicted class, matching
    the single element returned by the pipeline. A confidence of ``0.0`` is
    recorded when model output is malformed or empty — not treated as an error.
    """

    MODEL_ID: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
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

        Labels are title-cased: ``'Negative'``, ``'Neutral'``, ``'Positive'``.

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
