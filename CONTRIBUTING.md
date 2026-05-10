# Contributing to SYNAPSE Adapters

Thank you for contributing to the SYNAPSE adapter ecosystem. This document
explains the requirements for a contribution to be accepted.

## What is an adapter

An adapter is a pair of pure functions — ingress and egress — that connect a
real AI model to the SYNAPSE canonical IR. Write it once and your model is
immediately composable with every other registered model in the ecosystem.

## Before you start

Install the SDK: `pip install synapse-adapter-sdk`

Read the first adapter guide:
https://synapse-ir.github.io/adapter-sdk/getting-started/first-adapter/

Check BOUNTIES.md for models where adapters are most wanted. The first
contributor to a bounty model is listed as a Founding Contributor.

## Test policy

Every adapter must include tests. This is a hard requirement, not a suggestion.
As major new functionality is added to an adapter, tests of that functionality
must be added to the adapter's test suite. Pull requests that add adapter
functionality without corresponding tests will not be merged. This policy has
been followed for every adapter in this repository since the first commit.

Tests must:
- Use mock model output only — never call real models
- Cover the validator fixture suite (all 20 standard fixtures via
  `assert_valid_on`)
- Cover egress correctness, ingress correctness, and edge cases
- Not import model-specific libraries (torch, PIL, numpy) in test files

## Requirements for an acceptable adapter

Every pull request must pass all four of these checks before it will be merged.

### 1. All validator rules must pass

```
synapse-validate --adapter your_module.YourAdapter --all-fixtures
```

All 20 standard fixtures must pass with zero errors and zero warnings
(advisory warnings on soft payload size limits are acceptable).

### 2. Full test suite must pass

```
uv run pytest --tb=short -q
```

Your adapter must include a test file at
`your_model_folder/tests/test_your_model_adapter.py`. All existing tests must
continue to pass.

### 3. Ruff must find no violations

```
uv run ruff check .
```

### 4. Mypy must find no errors

```
uv run mypy your_model_folder/your_adapter.py
```

## Pre-PR checklist

Run this complete audit before opening a pull request:

```
uv run pytest --tb=short -q
uv run python -c "from synapse_sdk.testing import AdapterValidator; from your_adapter import YourAdapter; AdapterValidator(YourAdapter()).assert_valid(); print('Validator: all rules passed')"
uv run ruff check .
uv run mypy your_model_folder/your_adapter.py
```

All four must be clean before submitting.

## Adapter folder structure

Each adapter lives in its own folder:

```
your_model_name/
  your_model_adapter.py     — the adapter (ingress + egress)
  README.md                 — model details, schema, usage, license
  tests/
    test_your_model_adapter.py  — test suite using mock output only
```

The folder name should be short, lowercase, and underscore-separated
(e.g. `finbert`, `whisper_large_v3`, `clip_vit_base_patch32`).

## Adapter requirements

- `ingress` and `egress` must be pure functions — no network calls, no model
  loading, no side effects, no persistent state
- The model is called **outside** the adapter by the caller who owns it
- `ingress` prepares the input dict for the caller to pass to the model
- `egress` processes the model output and returns updated canonical IR
- `egress` must append exactly one `ProvenanceEntry` via
  `self.build_provenance()`
- `egress` must not modify any existing `ProvenanceEntry`
- `task_header` and `compliance_envelope` must be carried forward unchanged
- If the model produces PII-sensitive output, set `pii_present=True` in the
  compliance envelope

## Licensing

Your adapter code must be MIT licensed. If the underlying model requires a
commercial license (such as John Snow Labs Healthcare NLP), document this
prominently in the adapter README.

## Pull request process

1. Fork this repository
2. Create a branch: `git checkout -b feat/your-model-name`
3. Create your adapter folder and write the adapter, tests, and README
4. Run the full pre-PR checklist above
5. Open a pull request with a description that includes:
   - Which model the adapter wraps
   - Task types and domains it covers
   - Model license
   - Confirmation that all four checks pass

A maintainer will review within 14 days.

## Code of Conduct

All contributors are expected to follow the project
[Code of Conduct](CODE_OF_CONDUCT.md).

## Questions

Open an issue at https://github.com/synapse-ir/adapters/issues
