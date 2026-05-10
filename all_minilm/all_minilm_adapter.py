"""
SYNAPSE Adapter for sentence-transformers/all-MiniLM-L6-v2
===========================================================
Model:     sentence-transformers/all-MiniLM-L6-v2
Task:      Sentence embedding (384-dimensional vectors)
License:   Apache 2.0
Install:   pip install sentence-transformers
Spec:      https://github.com/synapse-ir/spec
"""

from __future__ import annotations

from typing import Any

import numpy as np

from synapse_sdk.base import AdapterBase
from synapse_sdk.types import CanonicalIR


class AllMiniLML6V2Adapter(AdapterBase):
    """
    Adapter for sentence-transformers/all-MiniLM-L6-v2.

    Encodes text as 384-dimensional dense floating-point vectors suitable for
    semantic similarity scoring, semantic search, and clustering. The model is
    a distilled MiniLM variant fine-tuned on sentence pairs; it is fast,
    lightweight (~80 MB), and fully open-source (Apache 2.0).

    Architecture — pure functions, caller-owned model
    -------------------------------------------------
    The adapter never loads or invokes the model. The caller owns the
    SentenceTransformer instance and calls model.encode(). The adapter
    provides two pure transformation steps:

      ingress  — converts CanonicalIR to the dict the caller passes to
                 model.encode()
      egress   — converts the numpy array returned by model.encode() back
                 into a CanonicalIR with an embedding payload

    Typical caller pattern::

        from sentence_transformers import SentenceTransformer
        from all_minilm_adapter import AllMiniLML6V2Adapter

        model   = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        adapter = AllMiniLML6V2Adapter()

        # 1. Prepare model input
        model_input  = adapter.ingress(ir)                      # {"sentences": ["..."]}

        # 2. Run the model (caller's responsibility)
        model_output = model.encode(model_input["sentences"])   # np.ndarray (1, 384)

        # 3. Convert output back to canonical IR
        result_ir = adapter.egress(model_output, ir, latency_ms=42)

        # 4. Access the embedding
        vector = result_ir.payload.vector   # list[float], length 384

    Confidence
    ----------
    Embedding models produce a complete, valid vector or raise an exception —
    there is no concept of partial confidence. Provenance confidence is fixed
    at 1.0 to reflect that any successfully returned embedding is fully valid.
    """

    MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    ADAPTER_VERSION: str = "1.0.0"
    EMBEDDING_DIM: int = 384  # all-MiniLM-L6-v2 always produces 384-dim vectors

    # -------------------------------------------------------------------------
    # ingress
    # -------------------------------------------------------------------------

    def ingress(self, ir: CanonicalIR) -> dict[str, Any]:
        """Return {"sentences": [text]} for the caller to pass to model.encode()."""
        content: str = ir.payload.content or ""
        return {"sentences": [content]}

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
        Convert a (1, 384) numpy array from model.encode() into a CanonicalIR
        with modality="embedding".

        If model_output is missing, wrong shape, or otherwise unusable, sets
        vector=[] and vector_dim=0 rather than raising.
        """
        vector: list[float]
        vector_dim: int

        try:
            if (
                isinstance(model_output, np.ndarray)
                and model_output.ndim == 2
                and model_output.shape[0] >= 1
                and model_output.shape[1] == self.EMBEDDING_DIM
            ):
                vector = model_output[0].tolist()
                vector_dim = self.EMBEDDING_DIM
            else:
                vector = []
                vector_dim = 0
        except Exception:  # noqa: BLE001
            vector = []
            vector_dim = 0

        updated = original_ir.clone()
        updated.payload.modality = "embedding"
        updated.payload.vector = vector
        updated.payload.vector_dim = vector_dim
        updated.payload.embedding_model = self.MODEL_ID

        updated.provenance.append(
            self.build_provenance(confidence=1.0, latency_ms=latency_ms)
        )

        return updated
