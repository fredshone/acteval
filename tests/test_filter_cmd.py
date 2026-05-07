"""Tests for acteval filter CLI subcommands."""

import pandas as pd
import pytest

from acteval.cli import _run_filter_cmd


_HOME_BASED = [
    {"pid": 0, "act": "home", "start": 0, "end": 8, "duration": 8},
    {"pid": 0, "act": "work", "start": 8, "end": 16, "duration": 8},
    {"pid": 0, "act": "home", "start": 16, "end": 24, "duration": 8},
]

_NOT_HOME_BASED_STARTS = [
    {"pid": 1, "act": "work", "start": 0, "end": 8, "duration": 8},
    {"pid": 1, "act": "home", "start": 8, "end": 24, "duration": 16},
]

_NOT_HOME_BASED_ENDS = [
    {"pid": 2, "act": "home", "start": 0, "end": 16, "duration": 16},
    {"pid": 2, "act": "work", "start": 16, "end": 24, "duration": 8},
]

_CONSEC_HOME = [
    {"pid": 3, "act": "home", "start": 0, "end": 8, "duration": 8},
    {"pid": 3, "act": "home", "start": 8, "end": 16, "duration": 8},
    {"pid": 3, "act": "work", "start": 16, "end": 20, "duration": 4},
    {"pid": 3, "act": "home", "start": 20, "end": 24, "duration": 4},
]

_CONSEC_SHOP = [
    {"pid": 4, "act": "home", "start": 0, "end": 8, "duration": 8},
    {"pid": 4, "act": "shop", "start": 8, "end": 12, "duration": 4},
    {"pid": 4, "act": "shop", "start": 12, "end": 16, "duration": 4},
    {"pid": 4, "act": "home", "start": 16, "end": 24, "duration": 8},
]

_ALL_ROWS = _HOME_BASED + _NOT_HOME_BASED_STARTS + _NOT_HOME_BASED_ENDS + _CONSEC_HOME + _CONSEC_SHOP


@pytest.fixture
def schedule_csv(tmp_path):
    p = tmp_path / "schedules.csv"
    pd.DataFrame(_ALL_ROWS).to_csv(p, index=False)
    return str(p)


@pytest.fixture
def empty_csv(tmp_path):
    p = tmp_path / "empty.csv"
    pd.DataFrame(columns=["pid", "act", "start", "end", "duration"]).to_csv(p, index=False)
    return str(p)


# ---------------------------------------------------------------------------
# non-home-based
# ---------------------------------------------------------------------------

def test_filter_non_home_based_stdout(schedule_csv, capsys):
    _run_filter_cmd(["non-home-based", schedule_csv])
    out = capsys.readouterr().out
    result = pd.read_csv(pd.io.common.StringIO(out))
    assert set(result["pid"].unique()) == {1, 2}
    assert 0 not in result["pid"].values
    assert 3 not in result["pid"].values


def test_filter_non_home_based_to_file(schedule_csv, tmp_path):
    out_path = str(tmp_path / "out.csv")
    _run_filter_cmd(["non-home-based", schedule_csv, "-o", out_path])
    result = pd.read_csv(out_path)
    assert set(result["pid"].unique()) == {1, 2}


def test_filter_non_home_based_empty(empty_csv, capsys):
    _run_filter_cmd(["non-home-based", empty_csv])
    out = capsys.readouterr().out
    result = pd.read_csv(pd.io.common.StringIO(out))
    assert len(result) == 0


def test_filter_non_home_based_missing_act_col(tmp_path):
    p = tmp_path / "no_act.csv"
    pd.DataFrame({"pid": [0], "start": [0], "end": [10], "duration": [10]}).to_csv(p, index=False)
    with pytest.raises(SystemExit):
        _run_filter_cmd(["non-home-based", str(p)])


# ---------------------------------------------------------------------------
# consecutive
# ---------------------------------------------------------------------------

def test_filter_consecutive_defaults(schedule_csv, capsys):
    _run_filter_cmd(["consecutive", schedule_csv])
    out = capsys.readouterr().out
    result = pd.read_csv(pd.io.common.StringIO(out))
    # pid 3 has consecutive home; pid 4 has consecutive shop (not in defaults)
    assert set(result["pid"].unique()) == {3}


def test_filter_consecutive_custom_act(schedule_csv, capsys):
    _run_filter_cmd(["consecutive", schedule_csv, "--act", "shop"])
    out = capsys.readouterr().out
    result = pd.read_csv(pd.io.common.StringIO(out))
    assert set(result["pid"].unique()) == {4}


def test_filter_consecutive_multiple_acts(schedule_csv, capsys):
    _run_filter_cmd(["consecutive", schedule_csv, "--act", "home", "shop"])
    out = capsys.readouterr().out
    result = pd.read_csv(pd.io.common.StringIO(out))
    assert set(result["pid"].unique()) == {3, 4}


def test_filter_consecutive_no_matches(schedule_csv, capsys):
    _run_filter_cmd(["consecutive", schedule_csv, "--act", "leisure"])
    out = capsys.readouterr().out
    result = pd.read_csv(pd.io.common.StringIO(out))
    assert len(result) == 0


def test_filter_consecutive_to_file(schedule_csv, tmp_path):
    out_path = str(tmp_path / "out.csv")
    _run_filter_cmd(["consecutive", schedule_csv, "-o", out_path])
    result = pd.read_csv(out_path)
    assert set(result["pid"].unique()) == {3}


def test_filter_consecutive_missing_act_col(tmp_path):
    p = tmp_path / "no_act.csv"
    pd.DataFrame({"pid": [0], "start": [0], "end": [10], "duration": [10]}).to_csv(p, index=False)
    with pytest.raises(SystemExit):
        _run_filter_cmd(["consecutive", str(p)])
