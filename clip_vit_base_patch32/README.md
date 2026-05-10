# clip-vit-base-patch32 Adapter

SYNAPSE adapter for [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32).

## Overview

This adapter enables multimodal zero-shot image classification using OpenAI's CLIP model.
CLIP (Contrastive Language–Image Pre-Training) jointly embeds images and text into a shared
latent space via contrastive learning. At inference time it ranks any set of natural-language
candidate labels against an image — no fine-tuning required.

The underlying task is **zero-shot image classification**: given an image and a list of text
descriptions, CLIP returns a ranked probability distribution over those descriptions.

## Model details

| Field | Value |
|-------|-------|
| Model ID | `openai/clip-vit-base-patch32` |
| Architecture | ViT-B/32 (Vision Transformer, patch size 32) |
| Training data | 400 million image-text pairs (WIT — WebImageText) |
| Training method | Contrastive learning (InfoNCE loss) |
| Embedding dim | 512 |
| License | MIT |
| HF page | https://huggingface.co/openai/clip-vit-base-patch32 |

## Verified output schema

The transformers `zero-shot-image-classification` pipeline returns a list of dicts sorted
by descending score. List length equals the number of candidate labels provided.

```python
from transformers import pipeline

clip = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
result = clip(
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    candidate_labels=["a photo of a cat", "a photo of a dog", "a photo of a car"],
)
# [
#   {'score': 0.9993917942047119, 'label': 'a photo of a cat'},
#   {'score': 0.0003519294841680676, 'label': 'a photo of a dog'},
#   {'score': 0.0002562698791734874, 'label': 'a photo of a car'},
# ]
```

Each dict has exactly two keys:
- `"label"` — the candidate label string
- `"score"` — softmax probability (float, sums to ~1.0 across all labels)

`result[0]` is always the top prediction. The adapter stores all labels in
`payload.labels` and uses `result[0].score` as the provenance confidence.

## Usage example

```python
from synapse_sdk.types import CanonicalIR, TaskHeader, Payload, ComplianceEnvelope, TaskType, Domain

ir = CanonicalIR(
    ir_version="1.0.0",
    message_id="...",
    task_header=TaskHeader(
        task_type=TaskType.classify,
        domain=Domain.general,
        candidate_labels=["a photo of a cat", "a photo of a dog", "a photo of a car"],
    ),
    payload=Payload(
        modality="text",
        content="http://images.cocodataset.org/val2017/000000039769.jpg",
    ),
    compliance_envelope=ComplianceEnvelope(),
)

# ingress → pipeline → egress
ingress_dict = adapter.ingress(ir)
# {"image": "http://...", "candidate_labels": ["a photo of a cat", ...]}

model_output = clip(**ingress_dict)
# [{"score": 0.9994, "label": "a photo of a cat"}, ...]

result_ir = adapter.egress(model_output, ir, latency_ms=120)
# result_ir.payload.labels[0].label  → "a photo of a cat"
# result_ir.payload.labels[0].score  → 0.9994
# result_ir.provenance[-1].confidence → 0.9994
```

## Multimodal note

This adapter handles an image + text input pair split across two IR fields:

- **`payload.content`** — the image input (file path string, URL, or PIL Image object)
- **`task_header.candidate_labels`** — the text candidate labels to rank against the image

If `candidate_labels` is `None` or empty, the adapter falls back to the default label set:
`["object", "animal", "vehicle", "person", "food"]`.

The ranked output (all labels with scores) is stored in `payload.labels` as a list of
`Classification` objects. `payload.content` is **not** modified by egress.

## Use cases

- **Visual search** — classify product images into taxonomy categories without per-category training
- **Content moderation** — detect whether an image depicts sensitive content using descriptive labels
- **Product recognition** — match images to free-text product descriptions
- **Dataset labelling** — quickly annotate large image collections with natural-language labels
- **Cross-modal retrieval** — rank images by text query similarity

## License

MIT. See [https://huggingface.co/openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32).
