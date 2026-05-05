# SYNAPSE Adapters

Community adapter collection for the SYNAPSE canonical IR ecosystem. Each adapter in this repository connects a real model to the ecosystem — write it once, interoperate with everything.

## Available adapters

| Adapter | Model | Task types | Domains | Language |
|---------|-------|------------|---------|----------|
| [NER BERT](adapters/ner_bert/) | dslim/bert-base-NER | extract | general, legal | Python |

## Contribute an adapter

1. Install the SDK: `pip install synapse-adapter-sdk`
2. Write your adapter following the [guide](https://docs.synapse-ir.io/adapters)
3. Validate: `synapse-validate --adapter your_module.YourAdapter --all-fixtures`
4. Open a PR to this repository

All adapters must pass the full 20-fixture validation suite before merge. See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete process.

## Adapter structure

```
adapters/
  your_model_name/
    adapter.py          # The adapter itself
    manifest.json       # Capability manifest for the registry
    tests/
      test_adapter.py   # Tests using mock model output
    README.md           # Usage instructions
```

## Bounties

We maintain a list of models we would like to see adapted. First contributor to write a validated adapter for a bounty model is listed as a Founding Contributor in the repository.

See [BOUNTIES.md](BOUNTIES.md) for the current list.

## Documentation

- [Writing adapters](https://docs.synapse-ir.io/adapters)
- [Adapter SDK](https://github.com/synapse-ir/adapter-sdk)
- [Canonical IR specification](https://github.com/synapse-ir/spec)

## License

MIT. Each adapter may carry the license of its original author. See individual adapter directories for details.
