"""Tests for the acteval CLI (_build_parser and _run)."""

import argparse
import os

import pandas as pd
import pytest

from acteval.cli import _build_parser, _run, _validate_schedule

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_OBSERVED_ROWS = [
    {"pid": 0, "act": "home", "start": 0, "end": 8, "duration": 8},
    {"pid": 0, "act": "work", "start": 8, "end": 16, "duration": 8},
    {"pid": 0, "act": "home", "start": 16, "end": 24, "duration": 8},
    {"pid": 1, "act": "home", "start": 0, "end": 10, "duration": 10},
    {"pid": 1, "act": "work", "start": 10, "end": 20, "duration": 10},
    {"pid": 1, "act": "home", "start": 20, "end": 24, "duration": 4},
]

_SYNTHETIC_ROWS = [
    {"pid": 0, "act": "home", "start": 0, "end": 9, "duration": 9},
    {"pid": 0, "act": "work", "start": 9, "end": 17, "duration": 8},
    {"pid": 0, "act": "home", "start": 17, "end": 24, "duration": 7},
    {"pid": 1, "act": "home", "start": 0, "end": 8, "duration": 8},
    {"pid": 1, "act": "work", "start": 8, "end": 16, "duration": 8},
    {"pid": 1, "act": "home", "start": 16, "end": 24, "duration": 8},
]


@pytest.fixture
def csv_files(tmp_path):
    """Write observed and synthetic schedule CSVs; return their paths."""
    obs_path = tmp_path / "observed.csv"
    syn_path = tmp_path / "synthetic.csv"
    pd.DataFrame(_OBSERVED_ROWS).to_csv(obs_path, index=False)
    pd.DataFrame(_SYNTHETIC_ROWS).to_csv(syn_path, index=False)
    return str(obs_path), str(syn_path)


@pytest.fixture
def attrs_csv(tmp_path):
    """Write a simple attrs CSV and return its path."""
    attrs = pd.DataFrame({"pid": [0, 1], "gender": ["m", "f"]})
    p = tmp_path / "attrs.csv"
    attrs.to_csv(p, index=False)
    return str(p)


def _make_args(target, model_pairs, **kwargs):
    """Build a minimal argparse.Namespace for _run."""
    defaults = dict(
        target=target,
        model=list(model_pairs),
        attrs=None,
        target_attrs=None,
        split_on=None,
        config=None,
        level="domains",
        output=None,
        verbose=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


def test_build_parser_returns_parser():
    p = _build_parser()
    assert isinstance(p, argparse.ArgumentParser)
    assert p.prog == "acteval"


def test_parser_requires_target_and_model():
    p = _build_parser()
    with pytest.raises(SystemExit):
        p.parse_args([])


def test_parser_minimal():
    p = _build_parser()
    args = p.parse_args(["obs.csv", "--model", "m1", "syn.csv"])
    assert args.target == "obs.csv"
    assert args.model == [["m1", "syn.csv"]]
    assert args.level == "domains"
    assert args.verbose is False


def test_parser_multiple_models():
    p = _build_parser()
    args = p.parse_args(
        ["obs.csv", "--model", "a", "a.csv", "--model", "b", "b.csv"]
    )
    assert args.model == [["a", "a.csv"], ["b", "b.csv"]]


def test_parser_level_choices():
    p = _build_parser()
    for level in ("domains", "groups", "features"):
        args = p.parse_args(["obs.csv", "--model", "m", "s.csv", "--level", level])
        assert args.level == level

    with pytest.raises(SystemExit):
        p.parse_args(["obs.csv", "--model", "m", "s.csv", "--level", "invalid"])


# ---------------------------------------------------------------------------
# _validate_schedule tests
# ---------------------------------------------------------------------------


def test_validate_schedule_all_columns_ok():
    df = pd.DataFrame(_OBSERVED_ROWS)
    _validate_schedule(df, "dummy.csv")  # should not raise or exit


def test_validate_schedule_start_end_only():
    df = pd.DataFrame(
        [{"pid": 0, "act": "home", "start": 0, "end": 8}]
    )
    _validate_schedule(df, "dummy.csv")  # start + end → no error


def test_validate_schedule_start_duration_only():
    df = pd.DataFrame(
        [{"pid": 0, "act": "home", "start": 0, "duration": 8}]
    )
    _validate_schedule(df, "dummy.csv")  # start + duration → no error


def test_validate_schedule_end_duration_only():
    df = pd.DataFrame(
        [{"pid": 0, "act": "home", "end": 8, "duration": 8}]
    )
    _validate_schedule(df, "dummy.csv")  # end + duration → no error


def test_validate_schedule_missing_pid_exits(capsys):
    df = pd.DataFrame([{"act": "home", "start": 0, "end": 8}])
    with pytest.raises(SystemExit) as exc:
        _validate_schedule(df, "bad.csv")
    assert exc.value.code != 0 or isinstance(exc.value.code, str)
    # Error message mentions missing column
    captured = capsys.readouterr()
    assert "pid" in captured.err or "pid" in str(exc.value.code)


def test_validate_schedule_missing_act_exits():
    df = pd.DataFrame([{"pid": 0, "start": 0, "end": 8}])
    with pytest.raises(SystemExit):
        _validate_schedule(df, "bad.csv")


def test_validate_schedule_only_one_timing_col_exits():
    df = pd.DataFrame([{"pid": 0, "act": "home", "start": 0}])
    with pytest.raises(SystemExit):
        _validate_schedule(df, "bad.csv")


def test_validate_schedule_no_timing_cols_exits():
    df = pd.DataFrame([{"pid": 0, "act": "home"}])
    with pytest.raises(SystemExit):
        _validate_schedule(df, "bad.csv")


# ---------------------------------------------------------------------------
# _run — success paths
# ---------------------------------------------------------------------------


def test_run_basic(csv_files, capsys):
    obs_path, syn_path = csv_files
    args = _make_args(obs_path, [["m1", syn_path]])
    _run(args)
    out = capsys.readouterr().out
    assert "Domain distances" in out
    assert "Best model" in out


def test_run_two_models(csv_files, capsys):
    obs_path, syn_path = csv_files
    args = _make_args(obs_path, [["m1", syn_path], ["m2", syn_path]])
    _run(args)
    out = capsys.readouterr().out
    assert "m1" in out
    assert "m2" in out


def test_run_start_end_only_input(tmp_path, capsys):
    """_run should work with schedules that only have start + end (no duration)."""
    rows = [
        {"pid": 0, "act": "home", "start": 0, "end": 8},
        {"pid": 0, "act": "work", "start": 8, "end": 16},
        {"pid": 1, "act": "home", "start": 0, "end": 12},
        {"pid": 1, "act": "work", "start": 12, "end": 24},
    ]
    p = tmp_path / "sched.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    args = _make_args(str(p), [["m1", str(p)]])
    _run(args)
    out = capsys.readouterr().out
    assert "Domain distances" in out


def test_run_saves_output(csv_files, tmp_path, capsys):
    obs_path, syn_path = csv_files
    out_dir = str(tmp_path / "results")
    args = _make_args(obs_path, [["m1", syn_path]], output=out_dir)
    _run(args)
    assert os.path.isdir(out_dir)
    out = capsys.readouterr().out
    assert "Results saved" in out


# ---------------------------------------------------------------------------
# _run — validation/error paths
# ---------------------------------------------------------------------------


def test_run_duplicate_model_name_exits(csv_files):
    obs_path, syn_path = csv_files
    args = _make_args(obs_path, [["m1", syn_path], ["m1", syn_path]])
    with pytest.raises(SystemExit):
        _run(args)


def test_run_attrs_name_not_in_model_exits(csv_files, attrs_csv):
    obs_path, syn_path = csv_files
    args = _make_args(obs_path, [["m1", syn_path]])
    args.attrs = [["unknown", attrs_csv]]
    with pytest.raises(SystemExit):
        _run(args)


def test_run_duplicate_attrs_name_exits(csv_files, attrs_csv):
    obs_path, syn_path = csv_files
    args = _make_args(obs_path, [["m1", syn_path]])
    args.attrs = [["m1", attrs_csv], ["m1", attrs_csv]]
    with pytest.raises(SystemExit):
        _run(args)


def test_run_split_on_without_target_attrs_exits(csv_files):
    obs_path, syn_path = csv_files
    args = _make_args(obs_path, [["m1", syn_path]])
    args.split_on = ["gender"]
    with pytest.raises(SystemExit):
        _run(args)


def test_run_target_attrs_without_split_on_exits(csv_files, attrs_csv):
    obs_path, syn_path = csv_files
    args = _make_args(obs_path, [["m1", syn_path]])
    args.target_attrs = attrs_csv
    with pytest.raises(SystemExit):
        _run(args)


def test_run_split_on_missing_model_attrs_exits(csv_files, attrs_csv):
    obs_path, syn_path = csv_files
    args = _make_args(obs_path, [["m1", syn_path]])
    args.target_attrs = attrs_csv
    args.split_on = ["gender"]
    # No per-model attrs supplied — should exit
    with pytest.raises(SystemExit):
        _run(args)


def test_run_target_schedule_missing_pid_exits(tmp_path, csv_files):
    _, syn_path = csv_files
    bad = tmp_path / "bad.csv"
    pd.DataFrame([{"act": "home", "start": 0, "end": 8}]).to_csv(bad, index=False)
    args = _make_args(str(bad), [["m1", syn_path]])
    with pytest.raises(SystemExit):
        _run(args)


def test_run_model_schedule_only_one_timing_col_exits(tmp_path, csv_files):
    obs_path, _ = csv_files
    bad = tmp_path / "bad.csv"
    pd.DataFrame([{"pid": 0, "act": "home", "start": 0}]).to_csv(bad, index=False)
    args = _make_args(obs_path, [["m1", str(bad)]])
    with pytest.raises(SystemExit):
        _run(args)
