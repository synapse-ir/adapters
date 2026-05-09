"""Pytest configuration — adds the adapters root to sys.path."""
import sys
from pathlib import Path

# Ensure the adapters directory is importable so test files can do
# `from ner_clinical_adapter import NerClinicalAdapter` without a package prefix.
_ROOT = str(Path(__file__).parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
