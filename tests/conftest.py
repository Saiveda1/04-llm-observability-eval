"""Shared pytest fixtures: a small deterministic dataset generated once."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmobs import analytics, generate  # noqa: E402

# A modest but non-trivial dataset: enough to exercise drift + detector.
N_TRACES = 120_000


@pytest.fixture(scope="session")
def dataset(tmp_path_factory) -> str:
    data_dir = tmp_path_factory.mktemp("obs_data")
    generate.generate(N_TRACES, data_dir, verbose=False)
    return str(data_dir)


@pytest.fixture(scope="session")
def con():
    return analytics.connect(threads=2)
