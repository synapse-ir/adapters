"""
SYNAPSE Adapter for Helsinki-NLP/opus-mt-en-fr
===============================================
Model:     Helsinki-NLP/opus-mt-en-fr
Task:      Neural machine translation — English to French (translate)
Domain:    multilingual
License:   Apache 2.0
Install:   pip install transformers torch sentencepiece
Spec:      https://github.com/synapse-ir/spec
"""

from __future__ import annotations

from typing import Any

from synapse_sdk import AdapterBase, CanonicalIR


class OpusMtEnFrAdapter(AdapterBase):
    """
    Adapter for Helsinki-NLP/opus-mt-en-fr.

    Translates English text to French using the Opus-MT seq2seq model trained
    by the Language Technology Research Group at the University of Helsinki.
    The model is based on the Marian NMT framework and was trained on OPUS
    parallel corpora. It produces fluent French output for general-domain text
    including news, conversational, and formal writing.

    Input / output format
    ---------------------
    ingress receives a CanonicalIR with ``payload.modality="text"`` and
    returns ``{"text": str}`` for the transformers translation pipeline.

    egress receives ``[{"translation_text": str}]`` — the list produced by the
    pipeline — and returns a CanonicalIR with ``payload.content`` set to the
    translated French string. The original English source text is replaced.

    Verified pipeline output schema::

        from transformers import pipeline
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
        result = translator("How are you?")
        # [{'translation_text': 'Comment allez-vous ?'}]

    The output is always a list with exactly one dict. The only key is
    ``"translation_text"`` whose value is a plain string. Seq2seq generative
    models do not emit a token-level confidence score.

    Architecture — pure functions, caller-owned pipeline
    ----------------------------------------------------
    The adapter never loads or invokes the model. The caller owns the
    transformers ``pipeline`` instance and drives inference. The adapter
    provides two pure transformation steps:

      ingress  — converts CanonicalIR to the dict the caller passes to the
                 pipeline
      egress   — converts ``[{"translation_text": "..."}]`` returned by the
                 pipeline back into a CanonicalIR

    Typical caller pattern::

        from transformers import pipeline
        from opus_mt_en_fr_adapter import OpusMtEnFrAdapter

        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
        adapter    = OpusMtEnFrAdapter()

        # 1. Prepare model input
        model_input = adapter.ingress(ir)
        # {"text": "How are you?"}

        # 2. Run the model (caller's responsibility)
        t0 = time.monotonic()
        model_output = translator(model_input["text"])
        latency_ms = int((time.monotonic() - t0) * 1000)
        # [{"translation_text": "Comment allez-vous ?"}]

        # 3. Convert output back to canonical IR
        result_ir = adapter.egress(model_output, ir, latency_ms=latency_ms)

        # 4. Access the translation — original content is REPLACED
        french_text = result_ir.payload.content

    Pattern generality — all Helsinki-NLP opus-mt-{src}-{tgt} models
    -----------------------------------------------------------------
    This adapter pattern works for ALL Helsinki-NLP opus-mt language-pair
    models (1000+ pairs). The transformers translation pipeline produces the
    same ``[{"translation_text": str}]`` output schema across the entire
    opus-mt family. To adapt a different language pair, copy this file and
    change only MODEL_ID. The ingress and egress logic is identical for every
    pair. Example model IDs: ``Helsinki-NLP/opus-mt-en-de``,
    ``Helsinki-NLP/opus-mt-en-es``, ``Helsinki-NLP/opus-mt-zh-en``,
    ``Helsinki-NLP/opus-mt-fr-en``, ``Helsinki-NLP/opus-mt-en-ru``.

    PII
    ---
    Translation does not extract person entities. ``compliance_envelope
    .pii_present`` is never upgraded to ``True`` by this adapter.

    Confidence
    ----------
    Seq2seq generative models do not emit a document-level confidence score —
    the model either produces a translation or raises an exception. Provenance
    confidence is fixed at ``1.0`` per the SYNAPSE contract for generative
    models. A failed run surfaces as an exception, not a low-confidence result.

    Payload mutation
    ----------------
    ``payload.content`` is REPLACED by the translated text — the original
    source string is not preserved in the returned IR. Store the source
    separately if you need it after the adapter call.
    """

    MODEL_ID: str = "Helsinki-NLP/opus-mt-en-fr"
    ADAPTER_VERSION: str = "1.0.0"

    # -------------------------------------------------------------------------
    # ingress
    # -------------------------------------------------------------------------

    def ingress(self, ir: CanonicalIR) -> dict[str, Any]:
        """
        Prepare the pipeline input dict from the canonical IR.

        Returns::

            {"text": str}   # payload.content; "" when content is None

        The caller passes ``result["text"]`` to the translation pipeline::

            translator(model_input["text"])
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
        Convert ``[{"translation_text": "..."}]`` from the transformers pipeline
        into a CanonicalIR with ``payload.content`` set to the French translation.

        ``payload.content`` REPLACES the original English source text — it is
        not appended. ``payload.entities`` and ``payload.labels`` are not
        touched.

        If ``model_output`` is an empty list, ``None``, a dict, or the
        ``"translation_text"`` key is absent or ``None``, ``payload.content``
        is set to ``""`` rather than raising.
        """
        translation: str

        try:
            if isinstance(model_output, list) and len(model_output) > 0:
                translation = str(model_output[0].get("translation_text", "") or "")
            else:
                translation = ""
        except Exception:  # noqa: BLE001
            translation = ""

        updated = original_ir.clone()
        updated.payload.content = translation

        updated.provenance.append(
            self.build_provenance(confidence=1.0, latency_ms=latency_ms)
        )

        return updated
