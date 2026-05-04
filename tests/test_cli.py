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


@pytest.fixture
def batch_dir(tmp_path):
    """Batch directory with two model subdirs, each containing only a schedule."""
    for name in ("model_a", "model_b"):
        subdir = tmp_path / name
        subdir.mkdir()
        pd.DataFrame(_SYNTHETIC_ROWS).to_csv(subdir / "schedule.csv", index=False)
    return str(tmp_path)


@pytest.fixture
def batch_dir_with_attrs(tmp_path):
    """Batch directory with two model subdirs, each containing a schedule and attrs."""
    for name in ("model_a", "model_b"):
        subdir = tmp_path / name
        subdir.mkdir()
        pd.DataFrame(_SYNTHETIC_ROWS).to_csv(subdir / "schedule.csv", index=False)
        pd.DataFrame({"pid": [0, 1], "gender": ["m", "f"]}).to_csv(
            subdir / "attrs.csv", index=False
        )
    return str(tmp_path)


def _make_args(target, model_pairs, **kwargs):
    """Build a minimal argparse.Namespace for _run."""
    defaults = dict(
        target=target,
        model=list(model_pairs) if model_pairs is not None else None,
        target_attrs=None,
        split_on=None,
        config=None,
        level="domains",
        output=None,
        verbose=False,
        batch=None,
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
    assert args.target_attrs is None
    assert args.model == [["m1", "syn.csv"]]
    assert args.level == "domains"
    assert args.verbose is False


def test_parser_target_attrs_positional():
    p = _build_parser()
    args = p.parse_args(["obs.csv", "ta.csv", "-m", "m1", "syn.csv"])
    assert args.target == "obs.csv"
    assert args.target_attrs == "ta.csv"


def test_parser_multiple_models():
    p = _build_parser()
    args = p.parse_args(["obs.csv", "--model", "a", "a.csv", "--model", "b", "b.csv"])
    assert args.model == [["a", "a.csv"], ["b", "b.csv"]]


def test_parser_level_choices():
    p = _build_parser()
    for level in ("domains", "groups", "features"):
        args = p.parse_args(["obs.csv", "--model", "m", "s.csv", "--level", level])
        assert args.level == level

    with pytest.raises(SystemExit):
        p.parse_args(["obs.csv", "--model", "m", "s.csv", "--level", "invalid"])


def test_parser_short_model_flag():
    p = _build_parser()
    args = p.parse_args(["obs.csv", "-m", "m1", "syn.csv"])
    assert args.target == "obs.csv"
    assert args.model == [["m1", "syn.csv"]]


def test_parser_model_inline_attrs():
    p = _build_parser()
    args = p.parse_args(["obs.csv", "-m", "m1", "syn.csv", "attrs.csv"])
    assert args.model == [["m1", "syn.csv", "attrs.csv"]]


def test_parser_short_flags():
    p = _build_parser()
    args = p.parse_args(
        ["obs.csv", "-m", "m", "s.csv", "-l", "groups", "-o", "out/", "-v"]
    )
    assert args.level == "groups"
    assert args.output == "out/"
    assert args.verbose is True


def test_parser_batch_flag():
    p = _build_parser()
    args = p.parse_args(["obs.csv", "--batch", "models/"])
    assert args.batch == "models/"
    assert args.model is None

    args2 = p.parse_args(["obs.csv", "-b", "models/"])
    assert args2.batch == "models/"


# ---------------------------------------------------------------------------
# _validate_schedule tests
# ---------------------------------------------------------------------------


def test_validate_schedule_all_columns_ok():
    df = pd.DataFrame(_OBSERVED_ROWS)
    _validate_schedule(df, "dummy.csv")  # should not raise or exit


def test_validate_schedule_start_end_only():
    df = pd.DataFrame([{"pid": 0, "act": "home", "start": 0, "end": 8}])
    _validate_schedule(df, "dummy.csv")  # start + end → no error


def test_validate_schedule_start_duration_only():
    df = pd.DataFrame([{"pid": 0, "act": "home", "start": 0, "duration": 8}])
    _validate_schedule(df, "dummy.csv")  # start + duration → no error


def test_validate_schedule_end_duration_only():
    df = pd.DataFrame([{"pid": 0, "act": "home", "end": 8, "duration": 8}])
    _validate_schedule(df, "dummy.csv")  # end + duration → no error


def test_validate_schedule_missing_pid_exits():
    df = pd.DataFrame([{"act": "home", "start": 0, "end": 8}])
    with pytest.raises(SystemExit) as exc:
        _validate_schedule(df, "bad.csv")
    assert "pid" in str(exc.value.code)


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


def test_run_inline_attrs(csv_files, attrs_csv, capsys):
    """3-arg -m form with target_attrs positional and split_on runs successfully."""
    obs_path, syn_path = csv_files
    args = _make_args(
        obs_path,
        [["m1", syn_path, attrs_csv]],
        target_attrs=attrs_csv,
        split_on=["gender"],
    )
    _run(args)
    out = capsys.readouterr().out
    assert "Domain distances" in out


def test_run_batch_basic(csv_files, batch_dir, capsys):
    obs_path, _ = csv_files
    args = _make_args(obs_path, None, batch=batch_dir)
    _run(args)
    out = capsys.readouterr().out
    assert "model_a" in out
    assert "model_b" in out
    assert "Domain distances" in out


def test_run_batch_with_attrs(csv_files, batch_dir_with_attrs, attrs_csv, capsys):
    """Batch discovery with attrs, target_attrs positional, and split_on runs successfully."""
    obs_path, _ = csv_files
    args = _make_args(
        obs_path,
        None,
        batch=batch_dir_with_attrs,
        target_attrs=attrs_csv,
        split_on=["gender"],
    )
    _run(args)
    out = capsys.readouterr().out
    assert "model_a" in out
    assert "attrs" in out


# ---------------------------------------------------------------------------
# _run — validation/error paths
# ---------------------------------------------------------------------------


def test_run_duplicate_model_name_exits(csv_files):
    obs_path, syn_path = csv_files
    args = _make_args(obs_path, [["m1", syn_path], ["m1", syn_path]])
    with pytest.raises(SystemExit):
        _run(args)


def test_run_split_on_without_target_attrs_exits(csv_files):
    obs_path, syn_path = csv_files
    args = _make_args(obs_path, [["m1", syn_path]])
    args.split_on = ["gender"]
    with pytest.raises(SystemExit):
        _run(args)


def test_run_target_attrs_without_split_on_exits(csv_files, attrs_csv):
    """All attrs present but --split-on absent → exit."""
    obs_path, syn_path = csv_files
    args = _make_args(obs_path, [["m1", syn_path, attrs_csv]], target_attrs=attrs_csv)
    with pytest.raises(SystemExit):
        _run(args)


def test_run_model_attrs_without_target_attrs_exits(csv_files, attrs_csv):
    """Model has inline attrs but TARGET_ATTRS not given → all-or-nothing violation."""
    obs_path, syn_path = csv_files
    args = _make_args(obs_path, [["m1", syn_path, attrs_csv]])
    with pytest.raises(SystemExit):
        _run(args)


def test_run_target_attrs_without_model_attrs_exits(csv_files, attrs_csv):
    """TARGET_ATTRS given but model has no attrs → all-or-nothing violation."""
    obs_path, syn_path = csv_files
    args = _make_args(obs_path, [["m1", syn_path]], target_attrs=attrs_csv)
    with pytest.raises(SystemExit):
        _run(args)


def test_run_split_on_missing_model_attrs_exits(csv_files, attrs_csv):
    """TARGET_ATTRS + split_on but no per-model attrs → all-or-nothing violation."""
    obs_path, syn_path = csv_files
    args = _make_args(
        obs_path, [["m1", syn_path]], target_attrs=attrs_csv, split_on=["gender"]
    )
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


def test_run_no_model_no_batch_exits(csv_files):
    obs_path, _ = csv_files
    args = _make_args(obs_path, None, batch=None)
    with pytest.raises(SystemExit):
        _run(args)


def test_run_batch_and_model_exits(csv_files, batch_dir):
    obs_path, syn_path = csv_files
    args = _make_args(obs_path, [["m1", syn_path]], batch=batch_dir)
    with pytest.raises(SystemExit):
        _run(args)


def test_run_model_spec_too_short_exits(csv_files):
    obs_path, _ = csv_files
    args = _make_args(obs_path, [["name_only"]])
    with pytest.raises(SystemExit):
        _run(args)


def test_run_model_spec_too_long_exits(csv_files):
    obs_path, syn_path = csv_files
    args = _make_args(obs_path, [["m1", syn_path, "attrs.csv", "extra_arg"]])
    with pytest.raises(SystemExit):
        _run(args)


def test_run_batch_no_schedule_exits(csv_files, tmp_path):
    """Model subdir with only an attrs file and no schedule → exit."""
    obs_path, _ = csv_files
    subdir = tmp_path / "model_a"
    subdir.mkdir()
    pd.DataFrame({"pid": [0, 1], "gender": ["m", "f"]}).to_csv(
        subdir / "attrs.csv", index=False
    )
    args = _make_args(obs_path, None, batch=str(tmp_path))
    with pytest.raises(SystemExit):
        _run(args)


def test_run_batch_ambiguous_schedule_exits(csv_files, tmp_path):
    """Model subdir with two schedule files → exit."""
    obs_path, _ = csv_files
    subdir = tmp_path / "model_a"
    subdir.mkdir()
    pd.DataFrame(_SYNTHETIC_ROWS).to_csv(subdir / "sched1.csv", index=False)
    pd.DataFrame(_SYNTHETIC_ROWS).to_csv(subdir / "sched2.csv", index=False)
    args = _make_args(obs_path, None, batch=str(tmp_path))
    with pytest.raises(SystemExit):
        _run(args)


def test_run_batch_inconsistent_attrs_exits(csv_files, tmp_path):
    """Some model subdirs have attrs and some do not → exit."""
    obs_path, _ = csv_files
    for i, name in enumerate(["model_a", "model_b"]):
        subdir = tmp_path / name
        subdir.mkdir()
        pd.DataFrame(_SYNTHETIC_ROWS).to_csv(subdir / "schedule.csv", index=False)
        if i == 0:
            pd.DataFrame({"pid": [0, 1], "gender": ["m", "f"]}).to_csv(
                subdir / "attrs.csv", index=False
            )
    args = _make_args(obs_path, None, batch=str(tmp_path))
    with pytest.raises(SystemExit):
        _run(args)
