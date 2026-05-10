"""
SYNAPSE Adapter for openai/whisper-large-v3
============================================
Model:     openai/whisper-large-v3
Task:      Automatic speech recognition — audio to text transcription (transcribe)
Domain:    audio, multilingual
License:   Apache 2.0
Install:   pip install transformers torch
Spec:      https://github.com/synapse-ir/spec
"""

from __future__ import annotations

from typing import Any

from synapse_sdk import AdapterBase, CanonicalIR


class WhisperLargeV3Adapter(AdapterBase):
    """
    Adapter for openai/whisper-large-v3.

    Transcribes speech audio to text using OpenAI's Whisper large-v3 model.
    Whisper is a seq2seq encoder-decoder transformer trained on 5 million hours
    of weakly supervised audio data, supporting automatic speech recognition (ASR)
    and translation across 99 languages. The large-v3 checkpoint achieves 10–20%
    word error rate (WER) improvement over large-v2 on standard benchmarks.

    Audio modality
    --------------
    The adapter expects audio input carried in ``ir.payload.content``. The
    transformers pipeline accepts:

    - a file path string (e.g. ``"/data/earnings_call.mp3"``)
    - a NumPy array of float32 samples at 16 kHz
    - a dict ``{"array": np.ndarray, "sampling_rate": int}``

    ingress passes the value through unchanged under the key ``"audio"``.

    Content-in / content-out pattern
    ---------------------------------
    ingress reads the audio reference from ``ir.payload.content``.
    egress writes the transcribed text back into ``updated.payload.content``,
    replacing the original audio reference with the plain transcription string.
    This mirrors the Helsinki-NLP/opus-mt content-in/content-out translation
    pattern.

    Verified pipeline output schema::

        from transformers import pipeline
        pipe = pipeline("automatic-speech-recognition",
                        model="openai/whisper-large-v3")
        result = pipe("audio.mp3")
        # {"text": " And so my fellow Americans..."}

    The output is always a single dict with one key: ``"text"``.  The value is
    the full transcription as a plain string.  Leading whitespace is common in
    Whisper output and is stripped by egress.

    Apache 2.0 license
    ------------------
    The model weights are released under the Apache 2.0 license. See
    https://huggingface.co/openai/whisper-large-v3 for the model card.

    Multilingual support
    --------------------
    Whisper large-v3 auto-detects the spoken language and supports transcription
    across 99 languages. No language tag is required in ingress; the model infers
    the language from the audio signal.

    PII
    ---
    Transcription reproduces spoken words verbatim but does not extract or
    identify named persons as entities. ``compliance_envelope.pii_present``
    is never upgraded to ``True`` by this adapter.

    Confidence
    ----------
    Whisper is a generative seq2seq model — it produces a transcription or raises
    an exception, never a token-level probability at the pipeline output level.
    Provenance confidence is fixed at ``1.0`` per the SYNAPSE contract for
    generative models.

    Payload mutation
    ----------------
    ``payload.content`` is REPLACED by the transcribed text — the original audio
    reference is not preserved in the returned IR. Store the source separately if
    you need it after the adapter call.
    """

    MODEL_ID: str = "openai/whisper-large-v3"
    ADAPTER_VERSION: str = "1.0.0"

    # -------------------------------------------------------------------------
    # ingress
    # -------------------------------------------------------------------------

    def ingress(self, ir: CanonicalIR) -> dict[str, Any]:
        """
        Prepare the pipeline input dict from the canonical IR.

        Returns::

            {"audio": <ir.payload.content>}

        ``ir.payload.content`` may be a file path string, a NumPy array, or a
        dict with ``{"array": ..., "sampling_rate": ...}``. It is passed through
        unchanged. When ``ir.payload.content`` is ``None``, ``""`` is used so
        that the caller receives a safe, non-null value.
        """
        audio = ir.payload.content if ir.payload.content is not None else ""
        return {"audio": audio}

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
        Convert ``{"text": "..."}`` from the transformers ASR pipeline into a
        CanonicalIR with ``payload.content`` set to the transcription string.

        ``payload.content`` REPLACES the original audio reference — it is not
        appended. ``payload.entities`` and ``payload.labels`` are not touched.

        If ``model_output`` is ``None``, not a dict, missing the ``"text"`` key,
        or the ``"text"`` value is ``None``, ``payload.content`` is set to ``""``
        rather than raising.  Non-string values under ``"text"`` are coerced with
        ``str()``.  Leading and trailing whitespace is always stripped.
        """
        transcription: str

        try:
            if isinstance(model_output, dict):
                raw = model_output.get("text")
                if raw is None:
                    transcription = ""
                else:
                    transcription = str(raw).strip()
            else:
                transcription = ""
        except Exception:  # noqa: BLE001
            transcription = ""

        updated = original_ir.clone()
        updated.payload.content = transcription

        updated.provenance.append(
            self.build_provenance(confidence=1.0, latency_ms=latency_ms)
        )

        return updated
