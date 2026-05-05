# SYNAPSE Adapter Bounties

First contributor to write a validated adapter for any model below
is listed as a Founding Contributor in this repository permanently.

## Open bounties

| Model | Task types | Domain | Notes |
|-------|-----------|--------|-------|
| [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) | classify | general | Zero-shot classification |
| [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER) | extract | general, legal | Reference implementation exists |
| [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) | classify, score | finance | Financial sentiment |
| [medicalai/ClinicalBERT](https://huggingface.co/medicalai/ClinicalBERT) | classify, extract | medical | Clinical NLP |
| [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base) | classify, embed | code | Code understanding |

## How to claim a bounty

1. Write a validated adapter following the [guide](https://docs.synapse-ir.io/adapters)
2. Run `synapse-validate --adapter your_module.YourAdapter --all-fixtures`
3. Open a PR to this repository
4. Tag your PR with the label `bounty`

All 13 validation rules must pass. All 20 standard fixtures must pass.
