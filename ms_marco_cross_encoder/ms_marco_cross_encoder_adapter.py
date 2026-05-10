"""
SYNAPSE Adapter for cross-encoder/ms-marco-MiniLM-L6-v2
========================================================
Model:     cross-encoder/ms-marco-MiniLM-L6-v2
Task:      Relevance scoring / reranking for RAG pipelines (rank)
Domain:    general
License:   Apache 2.0
Install:   pip install sentence-transformers torch
Spec:      https://github.com/synapse-ir/spec
"""

from __future__ import annotations

import math
from typing import Any

from synapse_sdk import AdapterBase, CanonicalIR


class MsMarcoCrossEncoderAdapter(AdapterBase):
    """
    Adapter for cross-encoder/ms-marco-MiniLM-L6-v2.

    Scores the relevance of a single passage with respect to a query using a
    cross-encoder architecture. The model was fine-tuned on the MS MARCO
    passage ranking dataset and is optimised for retrieval-augmented generation
    (RAG) reranking pipelines. It achieves NDCG@10 74.30 on TREC DL 2019 and
    MRR@10 39.01 on the MS MARCO Dev set.

    RAG reranking use case
    ----------------------
    In a RAG pipeline the typical flow is::

        1. Retrieve  — a dense or sparse retriever returns the top-k candidate
                        passages for the user's query.
        2. Rerank    — a cross-encoder (this model) scores each query+passage
                        pair and re-orders the candidates by relevance.
        3. Generate  — the reranked top passages are fed to an LLM as context.

    Cross-encoders jointly encode the query and the passage, which allows them
    to model fine-grained interactions that bi-encoders miss. The trade-off is
    that every query+passage pair requires a separate forward pass, so this
    adapter processes one pair at a time. Call it in a loop over candidates.

    Input format
    ------------
    ingress reads:
      - ``ir.task_header.query``   — the user query string (may be None → "")
      - ``ir.payload.content``     — the passage to score (may be None → "")

    egress receives:
      - a numpy.ndarray of shape (1,) OR a plain float / int

    Output format
    -------------
    The sentence-transformers ``CrossEncoder.predict()`` returns a raw logit
    (unbounded float, typically in the range -10 to +10). This raw score is
    NOT a probability. egress normalises it to [0.0, 1.0] using the sigmoid
    function and stores it in ``payload.score``.

    Sigmoid normalization
    ---------------------
    The raw logit is unbounded and can be negative. A sigmoid transform maps
    it to a principled [0.0, 1.0] probability-like score::

        score = 1 / (1 + exp(-raw_score))

    A raw score of 0 maps to exactly 0.5. Positive logits map above 0.5
    (relevant); negative logits map below 0.5 (not relevant). The score is
    clamped to [0.0, 1.0] after the transform as a floating-point safety
    measure.

    Architecture — pure functions, caller-owned model
    -------------------------------------------------
    The adapter never loads or invokes the model. The caller owns the
    ``CrossEncoder`` instance and drives inference. The adapter provides two
    pure transformation steps:

      ingress  — converts CanonicalIR to ``{"query": str, "passage": str}``
      egress   — converts the raw model output back into a CanonicalIR with
                 ``payload.score`` set to the sigmoid-normalised relevance

    Typical caller pattern::

        import time
        from sentence_transformers import CrossEncoder
        from ms_marco_cross_encoder_adapter import MsMarcoCrossEncoderAdapter

        model   = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
        adapter = MsMarcoCrossEncoderAdapter()

        # 1. Prepare model input
        model_input = adapter.ingress(ir)
        # {"query": "How many people live in Berlin?",
        #  "passage": "Berlin had a population of 3.7 million in 2022."}

        # 2. Run the model (caller's responsibility)
        t0 = time.monotonic()
        scores = model.predict([(model_input["query"], model_input["passage"])])
        latency_ms = int((time.monotonic() - t0) * 1000)
        # numpy.ndarray([8.607138], dtype=float32)

        # 3. Convert output back to canonical IR
        result_ir = adapter.egress(scores, ir, latency_ms=latency_ms)

        # 4. Access the relevance score in [0.0, 1.0]
        relevance = result_ir.payload.score

    PII
    ---
    Ranking does not extract person entities. ``compliance_envelope
    .pii_present`` is never upgraded to ``True`` by this adapter.

    Confidence
    ----------
    confidence equals the sigmoid-normalised relevance score. A perfectly
    irrelevant passage scores near 0.0; a highly relevant one near 1.0; the
    midpoint 0.5 corresponds to a raw logit of 0.0 (model is neutral).
    """

    MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    ADAPTER_VERSION: str = "1.0.0"

    # -------------------------------------------------------------------------
    # ingress
    # -------------------------------------------------------------------------

    def ingress(self, ir: CanonicalIR) -> dict[str, Any]:
        """
        Prepare the model input dict from the canonical IR.

        Returns::

            {
                "query":   str,  # ir.task_header.query;  "" when None
                "passage": str,  # ir.payload.content;    "" when None
            }

        The caller constructs the pair as::

            model.predict([(model_input["query"], model_input["passage"])])
        """
        return {
            "query":   ir.task_header.query or "",
            "passage": ir.payload.content or "",
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
        Convert the raw model output into a CanonicalIR with the relevance
        score normalised to [0.0, 1.0] stored in ``payload.score``.

        Accepted ``model_output`` forms:
          - ``numpy.ndarray`` — takes ``float(model_output[0])``
          - ``float`` or ``int`` — uses ``float(model_output)`` directly
          - Empty array, ``None``, ``str``, ``dict``, or other → raw_score = 0.0

        The raw score is passed through a sigmoid to yield a [0.0, 1.0] value.
        ``payload.content``, ``payload.entities``, and ``payload.labels`` are
        not modified.
        """
        raw_score: float = 0.0

        try:
            if model_output is None:
                raw_score = 0.0
            elif isinstance(model_output, (str, bytes, dict)):
                raw_score = 0.0
            elif isinstance(model_output, (int, float)):
                raw_score = float(model_output)
            elif hasattr(model_output, "__len__"):
                if len(model_output) == 0:
                    raw_score = 0.0
                else:
                    raw_score = float(model_output[0])
            else:
                raw_score = float(model_output)
        except Exception:  # noqa: BLE001
            raw_score = 0.0

        try:
            score = 1.0 / (1.0 + math.exp(-raw_score))
        except OverflowError:
            score = 0.0

        score = max(0.0, min(1.0, score))

        updated = original_ir.clone()
        updated.payload.score = score

        updated.provenance.append(
            self.build_provenance(confidence=score, latency_ms=latency_ms)
        )

        return updated
