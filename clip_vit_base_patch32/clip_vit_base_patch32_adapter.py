"""
SYNAPSE Adapter for openai/clip-vit-base-patch32
=================================================
Model:     openai/clip-vit-base-patch32
Task:      Zero-shot image classification (classify)
Domain:    multimodal, vision
License:   MIT
Install:   pip install transformers torch
Spec:      https://github.com/synapse-ir/spec

Multimodal zero-shot image classification via CLIP
--------------------------------------------------
CLIP (Contrastive Language–Image Pre-Training) jointly embeds images and text
into a shared latent space using contrastive learning on 400 million image-text
pairs scraped from the internet. At inference time the model computes cosine
similarity between the image embedding and each candidate label's text embedding,
then applies a softmax to produce a probability distribution — enabling zero-shot
classification over any arbitrary set of natural-language labels without
fine-tuning.

Image + text input pair
-----------------------
``ir.payload.content`` carries the image input: a file path string, a URL, or a
PIL Image object. ``ir.task_header.candidate_labels`` carries the candidate label
strings (e.g. ``["a photo of a cat", "a photo of a dog"]``). Both are required
for a meaningful inference.

CLIP contrastive training
-------------------------
During pre-training CLIP minimises the contrastive loss between paired (image,
text) embeddings in a shared 512-dimensional space. The ViT-B/32 image encoder
uses Vision Transformer patch size 32; the text encoder is a 63M-parameter
transformer. This architecture generalises to unseen label vocabularies at
inference time.

MIT license
-----------
The model weights are released under the MIT license. See
https://huggingface.co/openai/clip-vit-base-patch32 for the model card.
"""

from __future__ import annotations

from typing import Any

from synapse_sdk import AdapterBase, CanonicalIR
from synapse_sdk.types import Classification

_DEFAULT_LABELS: list[str] = ["object", "animal", "vehicle", "person", "food"]


class ClipVitBasePatch32Adapter(AdapterBase):
    """Adapter for openai/clip-vit-base-patch32 zero-shot image classification."""

    MODEL_ID: str = "openai/clip-vit-base-patch32"
    ADAPTER_VERSION: str = "1.0.0"

    # -------------------------------------------------------------------------
    # ingress
    # -------------------------------------------------------------------------

    def ingress(self, ir: CanonicalIR) -> dict[str, Any]:
        """
        Prepare the pipeline input dict from the canonical IR.

        Returns::

            {
                "image": str | PIL.Image,     # payload.content; "" when None
                "candidate_labels": list[str] # from task_header or default
            }
        """
        candidate_labels: list[str] = (
            ir.task_header.candidate_labels or _DEFAULT_LABELS
        )
        return {
            "image": ir.payload.content if ir.payload.content is not None else "",
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
        Convert the zero-shot image classification pipeline output into a CanonicalIR.

        Expected ``model_output`` schema::

            [
                {"score": 0.9993917942047119, "label": "a photo of a cat"},
                {"score": 0.0003519294841680676, "label": "a photo of a dog"},
                {"score": 0.0002562698791734874, "label": "a photo of a car"},
            ]

        A list of dicts sorted by descending score; each dict has exactly
        ``"score"`` (float) and ``"label"`` (str). All labels are stored in
        ``payload.labels`` as :class:`~synapse_sdk.types.Classification` objects.
        Confidence is ``result[0].score`` — the top label's probability.

        If ``model_output`` is ``None``, not a list, or empty, ``payload.labels``
        is set to ``[]`` and provenance confidence is ``0.0`` rather than raising.
        Non-dict items in the list are skipped gracefully.
        """
        classifications: list[Classification] = []
        confidence: float = 0.0

        try:
            if isinstance(model_output, list) and model_output:
                classifications = [
                    Classification(
                        label=str(item.get("label", "")),
                        score=float(item.get("score", 0.0)),
                    )
                    for item in model_output
                    if isinstance(item, dict)
                ]
                confidence = classifications[0].score if classifications else 0.0
        except Exception:  # noqa: BLE001
            pass

        updated = original_ir.clone()
        updated.payload.labels = classifications
        updated.provenance.append(
            self.build_provenance(confidence=confidence, latency_ms=latency_ms)
        )
        return updated
