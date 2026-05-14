## Summary

<!-- What adapter does this add or change? Link the related issue if one exists. -->

Closes #

## Checklist

- [ ] Adapter file follows the `{name}_adapter.py` naming convention
- [ ] Tests are in `{name}/tests/test_{name}_adapter.py`
- [ ] All tests pass: `uv run pytest {name}/`
- [ ] Ruff passes: `uv run ruff check .`
- [ ] Mypy passes: `uv run mypy {name}/{name}_adapter.py`
- [ ] Adapter validator passes: `uv run python -c "from synapse_sdk.testing import AdapterValidator; from {name}.{name}_adapter import {ClassName}; AdapterValidator({ClassName}()).assert_valid()"`
- [ ] Branch is rebased on latest `main`

## Test counts

<!-- e.g. "42 passed, 0 failed" -->

## Notes for reviewers

<!-- Anything unusual about the model, ingress/egress shape, or dependencies? -->
