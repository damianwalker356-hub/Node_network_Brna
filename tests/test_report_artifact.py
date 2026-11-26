from __future__ import annotations

from pathlib import Path

import pytest

from brna.reporting import load_records, render

ROOT=Path(__file__).parents[1]

@pytest.mark.req('R-7.3')
@pytest.mark.reporting
def test_committed_report_matches_results_records():
    expected=render(load_records(ROOT/'results/run_records'))
    assert (ROOT/'docs/verification_report.md').read_text()==expected
