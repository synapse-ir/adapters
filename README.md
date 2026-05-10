# SYNAPSE Adapters

Community adapter collection for the SYNAPSE canonical IR ecosystem. Each adapter in this repository connects a real model to the ecosystem — write it once, interoperate with everything.

## Available adapters

| Adapter | Model | Task types | Domains | Language | License |
|---------|-------|------------|---------|----------|---------|
| [NER BERT](ner_bert_adapter.py) | dslim/bert-base-NER | extract | general, legal | Python | MIT |
| [OpenAI Classifier](openai-classifier.ts) | openai/gpt-4o-mini | classify | general | TypeScript | MIT |
| [JSL Clinical NER](ner_clinical_adapter.py) | johnsnowlabs/ner_clinical | extract | medical | Python | MIT (adapter) — requires John Snow Labs Healthcare NLP license |
| [Docling](docling_adapter.py) | docling-project/docling | extract | document, general | Python | MIT |
| [all-MiniLM-L6-v2](all_minilm_adapter.py) | sentence-transformers/all-MiniLM-L6-v2 | embed | general | Python | Apache 2.0 |

## Contribute an adapter

1. Install the SDK: `pip install synapse-adapter-sdk`
2. Write your adapter following the [first adapter guide](https://synapse-ir.github.io/adapter-sdk/getting-started/first-adapter/)
3. Validate: `synapse-validate --adapter your_module.YourAdapter --all-fixtures`
4. Open a PR to this repository

All adapters must pass the full 20-fixture validation suite before merge. See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete process.

## Adapter structure

Each adapter is a single Python or TypeScript file at the root of this repository, accompanied by a test suite and optionally a dedicated README for models that require additional setup.

## Bounties

We maintain a list of models we would like to see adapted. The first contributor to write a validated adapter for a bounty model is listed as a Founding Contributor in the repository.

See [BOUNTIES.md](BOUNTIES.md) for the current list.

## Documentation

- [Writing your first adapter](https://synapse-ir.github.io/adapter-sdk/getting-started/first-adapter/)
- [Adapter validation rules](https://synapse-ir.github.io/adapter-sdk/getting-started/validation/)
- [Adapter SDK](https://github.com/synapse-ir/adapter-sdk)
- [Canonical IR specification](https://github.com/synapse-ir/spec)
- [Live pipeline demo](https://synapse-ir.github.io/adapter-sdk/demo.html)

## License

MIT. Each adapter is MIT licensed unless otherwise noted. Adapters for licensed models (such as John Snow Labs Healthcare NLP) require the underlying model's license separately. See individual adapter READMEs for details.
