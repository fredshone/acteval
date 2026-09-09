import pytest
from pandas import DataFrame

from acteval.evaluate import compare
from acteval.results import combine, compare_many


@pytest.fixture
def observed_b():
    """A second, slightly different target: three-person, more work time."""
    return DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 8, "duration": 8},
            {"pid": 0, "act": "work", "start": 8, "end": 16, "duration": 8},
            {"pid": 0, "act": "home", "start": 16, "end": 24, "duration": 8},
            {"pid": 1, "act": "home", "start": 0, "end": 6, "duration": 6},
            {"pid": 1, "act": "work", "start": 6, "end": 20, "duration": 14},
            {"pid": 1, "act": "home", "start": 20, "end": 24, "duration": 4},
        ]
    )


def test_combine_empty_raises(observed, synthetic):
    with pytest.raises(ValueError, match="at least one"):
        combine({})


def test_combine_non_evalresult_raises(observed, synthetic):
    with pytest.raises(TypeError, match="not an EvalResult"):
        combine({"a": compare(observed, synthetic), "b": "not a result"})


def test_combine_mismatched_split_shape_raises(observed, synthetic, observed_b):
    plain = compare(observed, synthetic)
    obs_attrs = DataFrame({"pid": [0, 1], "group": ["a", "b"]})
    synth_attrs = DataFrame({"pid": [0, 1], "group": ["a", "b"]})
    split = compare(
        observed_b,
        {"synthetic": synthetic},
        attributes={"synthetic": synth_attrs},
        target_attributes=obs_attrs,
        split_on=["group"],
    )
    with pytest.raises(ValueError, match="cannot mix results"):
        combine({"t1": plain, "t2": split})


def test_combine_namespaces_model_columns(observed, synthetic, observed_b):
    r1 = compare(observed, {"m": synthetic})
    r2 = compare(observed_b, {"m": synthetic})
    combined = combine({"t1": r1, "t2": r2})
    assert set(combined.model_names) == {"t1::m", "t2::m"}


def test_combine_summary_and_rank_models(observed, synthetic, observed_b):
    r1 = compare(observed, {"m": synthetic})
    r2 = compare(observed_b, {"m": synthetic})
    combined = combine({"t1": r1, "t2": r2})
    summary = combined.summary()
    assert set(summary.columns) == {"t1::m", "t2::m"}
    ranks = combined.rank_models()
    assert set(ranks.index) == {"t1::m", "t2::m"}
    assert combined.best_model in {"t1::m", "t2::m"}


def test_combine_preserves_splits(observed, synthetic, observed_b):
    obs_attrs = DataFrame({"pid": [0, 1], "group": ["a", "b"]})
    synth_attrs = DataFrame({"pid": [0, 1], "group": ["a", "b"]})
    r1 = compare(
        observed,
        {"m": synthetic},
        attributes={"m": synth_attrs},
        target_attributes=obs_attrs,
        split_on=["group"],
    )
    r2 = compare(
        observed_b,
        {"m": synthetic},
        attributes={"m": synth_attrs},
        target_attributes=obs_attrs,
        split_on=["group"],
    )
    combined = combine({"t1": r1, "t2": r2})
    assert combined.has_splits is True
    assert combined.domains.by_category.distances.index.names == [
        "domain",
        "label",
        "cat",
    ]
    assert "t1::m" in combined.domains.by_category.distances.columns
    assert "t2::m" in combined.domains.by_category.distances.columns


def test_combine_save(tmp_path, observed, synthetic, observed_b):
    r1 = compare(observed, {"m": synthetic})
    r2 = compare(observed_b, {"m": synthetic})
    combined = combine({"t1": r1, "t2": r2})
    combined.save(tmp_path)
    assert (tmp_path / "domains" / "distances.csv").exists()
    assert (tmp_path / "features" / "distances.csv").exists()


def test_compare_many_matches_manual_combine(observed, synthetic, observed_b):
    manual = combine(
        {
            "t1": compare(observed, {"m": synthetic}),
            "t2": compare(observed_b, {"m": synthetic}),
        }
    )
    via_helper = compare_many({"t1": observed, "t2": observed_b}, {"m": synthetic})
    assert manual.summary().equals(via_helper.summary())


def test_compare_many_with_evaluator_style_targets(observed, synthetic, observed_b):
    result = compare_many({"t1": observed, "t2": observed_b}, {"m": synthetic})
    assert set(result.model_names) == {"t1::m", "t2::m"}
    assert isinstance(result.rank_models().index[0], str)
