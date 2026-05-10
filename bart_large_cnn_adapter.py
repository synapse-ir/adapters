"""
SYNAPSE Adapter for facebook/bart-large-cnn
============================================
Model:     facebook/bart-large-cnn
Task:      Abstractive text summarization (CNN/DailyMail)
License:   MIT
Install:   pip install transformers torch
Spec:      https://github.com/synapse-ir/spec
"""

from __future__ import annotations

from typing import Any

from synapse_sdk.base import AdapterBase
from synapse_sdk.types import CanonicalIR


class BartLargeCNNAdapter(AdapterBase):
    """
    Adapter for facebook/bart-large-cnn.

    Produces abstractive summaries of English-language text. The model was
    fine-tuned on the CNN/DailyMail dataset and excels at summarising news
    articles, reports, and similarly structured prose. It generates fluent,
    novel sentences — not extracts — so the summary may paraphrase rather
    than quote the source.

    1024-token hard limit
    ---------------------
    BART's encoder accepts at most 1024 tokens (~3 000–4 000 characters of
    typical English text). Text that exceeds this limit is silently truncated
    by the transformers pipeline; the summary will reflect only the leading
    portion of the document.

    Chunking is the caller's responsibility. This adapter intentionally does
    NOT split long text internally — doing so would require multiple model
    invocations and belongs in the orchestration layer, not the adapter.

    Architecture — pure functions, caller-owned pipeline
    ----------------------------------------------------
    The adapter never loads or invokes the model. The caller owns the
    transformers ``pipeline`` instance and drives inference. The adapter
    provides two pure transformation steps:

      ingress  — converts CanonicalIR to the dict the caller passes to the
                 pipeline (text string + generation kwargs)
      egress   — converts ``[{"summary_text": "..."}]`` returned by the
                 pipeline back into a CanonicalIR

    Typical caller pattern::

        from transformers import pipeline
        from bart_large_cnn_adapter import BartLargeCNNAdapter

        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        adapter    = BartLargeCNNAdapter()

        # 1. Prepare model input
        model_input = adapter.ingress(ir)
        # {"text": "...", "max_length": 130, "min_length": 30, "do_sample": False}

        # 2. Run the model (caller's responsibility)
        t0 = time.monotonic()
        model_output = summarizer(
            model_input["text"],
            max_length=model_input["max_length"],
            min_length=model_input["min_length"],
            do_sample=model_input["do_sample"],
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        # [{"summary_text": "Generated summary here."}]

        # 3. Convert output back to canonical IR
        result_ir = adapter.egress(model_output, ir, latency_ms=latency_ms)

        # 4. Access the summary — original content is REPLACED, not appended
        summary = result_ir.payload.content

    Confidence
    ----------
    Generative models such as BART do not emit a document-level confidence
    score — the model either produces a summary or raises an exception. There
    is no partial or probabilistic confidence value. Provenance confidence is
    fixed at 1.0 to reflect that any successfully returned summary is fully
    valid; a failed run surfaces as an exception, not a low-confidence result.

    Payload mutation
    ----------------
    ``payload.content`` is REPLACED by the summary text — the original
    source document is not preserved in the returned IR. Store the source
    separately if you need it after the adapter call.
    """

    MODEL_ID: str = "facebook/bart-large-cnn"
    ADAPTER_VERSION: str = "1.0.0"

    DEFAULT_MAX_LENGTH: int = 130  # maximum summary length in tokens
    DEFAULT_MIN_LENGTH: int = 30   # minimum summary length in tokens

    # -------------------------------------------------------------------------
    # ingress
    # -------------------------------------------------------------------------

    def ingress(self, ir: CanonicalIR) -> dict[str, Any]:
        """
        Prepare the pipeline input dict from the canonical IR.

        Returns::

            {
                "text":       str,   # source text; "" when payload.content is None
                "max_length": 130,   # token budget for the generated summary
                "min_length": 30,    # minimum tokens the summary must reach
                "do_sample":  False, # deterministic greedy/beam output
            }

        The caller passes these to the pipeline as::

            summarizer(
                input["text"],
                max_length=input["max_length"],
                min_length=input["min_length"],
                do_sample=input["do_sample"],
            )

        HARD LIMIT: BART accepts at most 1024 tokens (~3 000–4 000 characters).
        Text longer than this will be silently truncated by the transformers
        pipeline. The caller is responsible for chunking long documents before
        invoking the pipeline. Do NOT chunk inside this adapter.
        """
        content: str = ir.payload.content or ""
        return {
            "text": content,
            "max_length": self.DEFAULT_MAX_LENGTH,
            "min_length": self.DEFAULT_MIN_LENGTH,
            "do_sample": False,
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
        Convert ``[{"summary_text": "..."}]`` from the transformers pipeline
        into a CanonicalIR with ``payload.content`` set to the summary.

        ``payload.content`` REPLACES the original source text — it is not
        appended. ``payload.content_length`` is set to the character count of
        the summary.

        If ``model_output`` is an empty list or the ``"summary_text"`` key is
        absent, ``payload.content`` is set to ``""`` and
        ``payload.content_length`` to ``0`` rather than raising.
        """
        summary_text: str

        try:
            if isinstance(model_output, list) and len(model_output) > 0:
                summary_text = str(model_output[0].get("summary_text", ""))
            else:
                summary_text = ""
        except Exception:  # noqa: BLE001
            summary_text = ""

        updated = original_ir.clone()
        updated.payload.modality = "text"
        updated.payload.content = summary_text
        updated.payload.content_length = len(summary_text)

        updated.provenance.append(
            self.build_provenance(confidence=1.0, latency_ms=latency_ms)
        )

        return updated
