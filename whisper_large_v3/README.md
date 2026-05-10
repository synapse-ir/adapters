# whisper-large-v3 SYNAPSE Adapter

SYNAPSE adapter for [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) — automatic speech recognition across 99 languages.

## Overview

Whisper large-v3 is OpenAI's state-of-the-art automatic speech recognition (ASR) model. It converts audio input — earnings calls, analyst briefings, interviews, or any spoken content — into a plain text transcription. The model handles diverse accents, noisy environments, and mixed-language audio with high accuracy across 99 languages.

This adapter implements the **audio modality** with a **content-in / content-out** pattern: the audio reference arrives in `payload.content` and the transcription is written back into `payload.content` on the output IR.

## Model details

| Field | Value |
|-------|-------|
| Model ID | `openai/whisper-large-v3` |
| Task | Automatic speech recognition (transcribe) |
| Architecture | Seq2seq encoder-decoder transformer |
| Training data | 5 million+ hours of weakly supervised audio |
| WER improvement | 10–20% over large-v2 on standard benchmarks |
| Languages | 99 languages with automatic language detection |
| License | Apache 2.0 |
| HuggingFace | https://huggingface.co/openai/whisper-large-v3 |

## Verified output schema

The transformers pipeline returns a single dict with one key:

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
result = pipe("audio.mp3")
# {"text": " And so my fellow Americans..."}
```

The `"text"` value is the full transcription as a plain string. Leading whitespace is common in Whisper output; the adapter strips it automatically. There is no per-token confidence score — Whisper is a generative model.

## Usage example

```python
import time
from transformers import pipeline
from whisper_large_v3_adapter import WhisperLargeV3Adapter

pipe    = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
adapter = WhisperLargeV3Adapter()

# 1. Prepare model input from the canonical IR
model_input = adapter.ingress(ir)
# {"audio": "/data/earnings_call.mp3"}

# 2. Run the model (caller's responsibility)
t0 = time.monotonic()
model_output = pipe(model_input["audio"])
latency_ms = int((time.monotonic() - t0) * 1000)
# {"text": " Good morning. Revenue for Q3 came in at..."}

# 3. Convert output back to canonical IR
result_ir = adapter.egress(model_output, ir, latency_ms=latency_ms)

# 4. Access the transcription — original audio reference is REPLACED
transcript = result_ir.payload.content
# "Good morning. Revenue for Q3 came in at..."
```

## Audio input formats

The `ingress` method passes `ir.payload.content` through unchanged under the `"audio"` key. The transformers pipeline accepts three input formats:

| Format | Example |
|--------|---------|
| File path string | `"/data/call.mp3"` |
| NumPy float32 array at 16 kHz | `np.array([...], dtype=np.float32)` |
| Dict with array and sampling rate | `{"array": np.ndarray, "sampling_rate": 16000}` |

Common audio formats supported: MP3, WAV, FLAC, OGG, M4A.

## Multilingual note

Whisper large-v3 automatically detects the spoken language from the audio signal — no language tag is required in the IR. The model supports 99 languages and achieves near-human accuracy on major European, Asian, and Middle Eastern languages. To force a specific language or enable translation to English, pass `generate_kwargs` to the pipeline (caller's responsibility, not part of the SYNAPSE adapter contract).

## Finance / enterprise use cases

Whisper large-v3 is well-suited for financial and enterprise audio processing:

- **Earnings calls** — transcribe quarterly earnings calls for downstream NLP analysis
- **Analyst briefings** — convert recorded analyst Q&A sessions to searchable text
- **Board meetings** — produce verbatim minutes from audio recordings
- **Compliance recordings** — transcribe regulatory call recordings for audit trails

Pair with the FinBERT or BART-MNLI adapters to classify or analyze the transcribed text within the same SYNAPSE pipeline.

## License

The adapter is MIT licensed. The underlying model weights (`openai/whisper-large-v3`) are released under the **Apache 2.0** license. See the [model card](https://huggingface.co/openai/whisper-large-v3) for full details.
