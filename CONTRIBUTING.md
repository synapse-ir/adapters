# Contributing to SYNAPSE Adapters

Thank you for contributing to the SYNAPSE adapter ecosystem. This document explains the requirements for a contribution to be accepted.

## What is an adapter

An adapter is a pair of pure functions — ingress and egress — that connect a real AI model to the SYNAPSE canonical IR. Write it once and your model is immediately composable with every other registered model in the ecosystem.

## Before you start

Install the SDK: pip install synapse-adapter-sdk

Read the first adapter guide: https://synapse-ir.github.io/adapter-sdk/getting-started/first-adapter/

Check BOUNTIES.md for models where adapters are most wanted. First contributor to a bounty model is listed as a Founding Contributor.

## Requirements for an acceptable adapter

Every pull request must pass all four of these checks before it will be merged.

### 1. All 13 validator rules must pass

Run: synapse-validate --adapter your_module.YourAdapter --all-fixtures

All 20 standard fixtures must pass with zero errors and zero warnings (advisory warnings on soft payload size limits are acceptable).

### 2. Full test suite must pass

Run: uv run pytest tests/ -v --tb=short

Your adapter must include a test file at tests/test_your_adapter.py. Tests must use mock model output — do not call real models in tests. All existing tests must continue to pass.

### 3. Ruff must find no violations

Run: uv run ruff check .

### 4. Mypy must find no errors

Run: uv run mypy your_adapter.py

## Pre-PR checklist

Run this complete audit before opening a pull request:

uv run pytest tests/ -v --tb=short
uv run python -c "from synapse_sdk.testing import AdapterValidator; from your_adapter import YourAdapter; AdapterValidator(YourAdapter()).assert_valid(); print('Validator: all rules passed')"
uv run ruff check .
uv run mypy your_adapter.py

All four must be clean before submitting.

## Adapter structure

Each adapter consists of:

- your_model_adapter.py — the adapter itself (ingress + egress)
- tests/test_your_model_adapter.py — test suite using mock output
- your_model_adapter_README.md — required for models with licensing requirements or non-obvious setup; recommended for all adapters

## Adapter requirements

- ingress and egress must be pure functions — no network calls, no model loading, no side effects, no persistent state
- The model is called OUTSIDE the adapter by the caller who owns it
- ingress prepares the input dict for the caller to pass to the model
- egress processes the model output and returns updated canonical IR
- egress must append exactly one ProvenanceEntry via self.build_provenance()
- egress must not modify any existing ProvenanceEntry
- task_header and compliance_envelope must be carried forward unchanged
- If the model produces PII-sensitive output, set pii_present=True in the compliance envelope

## Licensing

Your adapter code must be MIT licensed. If the underlying model requires a commercial license (such as John Snow Labs Healthcare NLP), document this prominently in the adapter README.

## Pull request process

1. Fork this repository
2. Create a branch: git checkout -b feat/your-model-name
3. Write your adapter and tests
4. Run the full pre-PR checklist above
5. Open a pull request with a description that includes which model the adapter wraps, the task types and domains it covers, the model license, and confirmation that all four checks pass

A maintainer will review within 14 days.

## Questions

Open an issue at https://github.com/synapse-ir/adapters/issues
