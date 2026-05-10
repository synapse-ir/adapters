# SYNAPSE Adapter Bounties

First contributor to write a validated adapter for any model below
is listed as a Founding Contributor in this repository permanently.

## Open bounties

| Model | Task | Domain | Notes | Difficulty |
|-------|------|--------|-------|------------|
| [deepset/roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2) | extract | general | Question answering / reading comprehension | Beginner |
| [medicalai/ClinicalBERT](https://huggingface.co/medicalai/ClinicalBERT) | classify | medical | Clinical text classification | Beginner |
| [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base) | embed | code | Code understanding and search | Intermediate |
| [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | embed | multilingual | Multilingual sentence embeddings | Beginner |
| [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) | summarize | general | Instruction-tuned summarization | Intermediate |
| [yiyanghkust/finbert-tone](https://huggingface.co/yiyanghkust/finbert-tone) | classify | finance | Financial tone detection | Beginner |
| [nlptown/bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) | classify | multilingual | Multilingual sentiment 1-5 stars | Beginner |
| [Jean-Baptiste/roberta-large-ner-english](https://huggingface.co/Jean-Baptiste/roberta-large-ner-english) | extract | general | Higher-accuracy NER alternative to bert-base-NER | Intermediate |

## How to claim a bounty

1. Check that the bounty is still open (no merged PR for this model yet)
2. Comment on the relevant issue to claim it so others know it is in progress
3. Write a validated adapter following the [first adapter guide](https://synapse-ir.github.io/adapter-sdk/getting-started/first-adapter/)
4. Run the full audit:

   uv run pytest your_model/tests/ -v --tb=short
   synapse-validate --adapter your_module.YourAdapter --all-fixtures
   uv run ruff check your_model/
   uv run mypy your_model/your_adapter.py

5. Open a PR to this repository and tag it with the bounty label

All validator rules must pass. All 20 standard fixtures must pass.
Founding Contributor credit is permanent and listed in this file once your PR merges.

## Claimed bounties

None yet — all bounties above are open.

## Completed bounties

None yet.
