# SYNAPSE Adapters

Community adapter collection for the SYNAPSE canonical IR ecosystem.
Each adapter in this repository connects a real model to the ecosystem —
write it once, interoperate with everything.

## Available adapters

| Adapter | Model | Task types | Domains | Language |
|---------|-------|------------|---------|----------|
| [NER BERT](adapters/ner_bert/) | dslim/bert-base-NER | extract | general, legal | Python |

## Contribute an adapter

1. Install the SDK: `pip install synapse-adapter-sdk`
2. Write your adapter following the [guide](https://docs.synapse-ir.io/adapters)
3. Validate: `synapse-validate --adapter your_module.YourAdapter --all-fixtures`
4. Open a PR to this repository

All adapters must pass the full 20-fixture validation suite before merge.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete process.

## Adapter structure
