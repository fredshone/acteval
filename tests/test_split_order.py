"""Tests for n-th order attribute slicing (split_order) and containment views."""

import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from acteval._pipeline import _aggregate_features
from acteval._result_frame import ResultFrame
from acteval._splits import Slice, build_slices
from acteval.cli import _build_parser, _run
from acteval.evaluate import Evaluator, compare
from acteval.results import combine

sys.path.insert(0, str(Path(__file__).parent))
from test_bench_evaluate import generate_population  # noqa: E402

ATTRS = {"gender": ["m", "f"], "income": ["low", "mid", "high"], "car": ["y", "n"]}


def _attrs(df, seed):
    rng = np.random.default_rng(seed)
    pids = df.pid.unique()
    a = DataFrame({"pid": pids})
    for col, values in ATTRS.items():
        a[col] = rng.choice(values, size=len(pids))
    return a


@pytest.fixture(scope="module")
def pops():
    obs = generate_population(3_000, seed=1)
    syn = generate_population(3_000, seed=2)
    return obs, syn, _attrs(obs, 1), _attrs(syn, 2)


@pytest.fixture(scope="module")
def order2(pops):
    obs, syn, ta, sa = pops
    return compare(
        obs,
        {"m": syn},
        target_attributes=ta,
        synthetic_attributes={"m": sa},
        split_on=list(ATTRS),
        split_order=2,
        min_split_size=0,
    )


# --- build_slices ------------------------------------------------------------


def test_build_slices_counts_and_keys():
    attrs = DataFrame({"pid": range(6), "a": list("xxyyzz"), "b": [1, 2, 1, 2, 1, 2]})
    order1 = build_slices(attrs, ["a", "b"], 1)
    assert [s.key for s, _ in order1] == [
        ("a", "x"),
        ("a", "y"),
        ("a", "z"),
        ("b", 1),
        ("b", 2),
    ]
    order2 = build_slices(attrs, ["a", "b"], 2, min_size=0)
    assert len(order2) == 6
    s, pids = order2[0]
    assert s == Slice(attrs=("a", "b"), values=("x", 1))
    assert s.key == ("a&b", "x&1")
    assert s.assignment == (("a", "x"), ("b", 1))
    np.testing.assert_array_equal(pids, [0])


def test_build_slices_min_size_prunes_and_warns():
    attrs = DataFrame({"pid": range(5), "a": list("xxxxy"), "b": list("ppppp")})
    with pytest.warns(UserWarning, match="skipped 1 of 2"):
        slices = build_slices(attrs, ["a", "b"], 2, min_size=2)
    assert [s.values for s, _ in slices] == [("x", "p")]


def test_build_slices_default_min_size_only_above_order1():
    attrs = DataFrame({"pid": range(4), "a": list("xxyy"), "b": list("pqpq")})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(build_slices(attrs, ["a", "b"], 1)) == 4
    with pytest.raises(ValueError, match="fewer than min_split_size=30"):
        build_slices(attrs, ["a", "b"], 2)


def test_build_slices_order_out_of_range():
    attrs = DataFrame({"pid": [0], "a": ["x"]})
    with pytest.raises(ValueError, match="between 1 and"):
        build_slices(attrs, ["a"], 2)
    with pytest.raises(ValueError, match="between 1 and"):
        build_slices(attrs, ["a"], 0)


# --- pipeline ------------------------------------------------------------------


def test_order1_matches_default(pops):
    obs, syn, ta, sa = pops
    kwargs = dict(
        target_attributes=ta, synthetic_attributes={"m": sa}, split_on=list(ATTRS)
    )
    default = compare(obs, {"m": syn}, **kwargs)
    explicit = compare(obs, {"m": syn}, split_order=1, **kwargs)
    pd.testing.assert_frame_equal(
        default.raw["distances"].values, explicit.raw["distances"].values
    )
    assert default.split_order == 1


def test_order2_matches_combined_column_split(pops, order2):
    """split_order=2 must equal order-1 splitting on pre-joined pair columns."""
    obs, syn, ta, sa = pops

    def joined(a):
        a = a.copy()
        for c1, c2 in itertools.combinations(ATTRS, 2):
            a[f"{c1}&{c2}"] = a[c1] + "&" + a[c2]
        return a

    pair_cols = [f"{c1}&{c2}" for c1, c2 in itertools.combinations(ATTRS, 2)]
    emulated = compare(
        obs,
        {"m": syn},
        target_attributes=joined(ta),
        synthetic_attributes={"m": joined(sa)},
        split_on=pair_cols,
    )
    for kind in ("descriptions", "distances"):
        for part in ("values", "weights"):
            a = getattr(order2.raw[kind], part).sort_index()
            b = getattr(emulated.raw[kind], part).sort_index()
            pd.testing.assert_frame_equal(a, b)


def test_order2_labels(order2):
    labels = set(order2.raw["distances"].values.index.get_level_values("label"))
    assert labels == {"gender&income", "gender&car", "income&car"}
    assert order2.split_order == 2
    assert order2.slice_assignments[("gender&income", "m&high")] == (
        ("gender", "m"),
        ("income", "high"),
    )


def test_split_order_without_split_on_raises(observed):
    with pytest.raises(ValueError, match="requires split_on"):
        Evaluator(observed, split_order=2)


def test_config_split_order(tmp_path, pops):
    obs, syn, ta, sa = pops
    default = Path(__file__).parents[1] / "src/acteval/config.toml"
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        default.read_text().replace("order = 1\n", "order = 2\nmin_size = 0\n")
    )
    ev = Evaluator(obs, target_attributes=ta, split_on=list(ATTRS), config_path=cfg)
    assert ev._split_order == 2
    # Ignored (not an error) without split_on.
    assert Evaluator(obs, config_path=cfg)._split_order == 1


# --- views ---------------------------------------------------------------------


def test_default_views_are_computed_order(order2):
    labels = order2.at("groups", "by_attribute").distances.index.get_level_values(
        "label"
    )
    assert set(labels) == {"gender&income", "gender&car", "income&car"}


def test_order1_by_attribute_view_labels(order2):
    view = order2.at("domains", "by_attribute", order=1).distances
    assert set(view.index.get_level_values("label")) == set(ATTRS)
    cats = order2.at("domains", "by_category", order=1).distances
    assert ("m" in cats.index.get_level_values("cat")) and (
        "high" in cats.index.get_level_values("cat")
    )


def test_order1_category_view_pools_containing_slices(order2):
    """gender=m at order 1 == weighted pool of every gender=m & * slice."""
    dist = order2.raw["distances"]
    desc = order2.raw["descriptions"]
    idx = dist.values.index
    keep = [
        ("gender", "m") in order2.slice_assignments[(label, cat)]
        for label, cat in zip(
            idx.get_level_values("label"), idx.get_level_values("cat")
        )
    ]
    # Each gender=m row appears once per order-1 sub-key ("gender=m"), so the
    # pool is just those rows aggregated with the label/cat levels dropped.
    sub = ResultFrame(
        values=dist.values[keep], weights=dist.weights[keep], units=dist.units[keep]
    )
    dkeep = [
        ("gender", "m") in order2.slice_assignments[(label, cat)]
        for label, cat in zip(
            desc.values.index.get_level_values("label"),
            desc.values.index.get_level_values("cat"),
        )
    ]
    sub_desc = ResultFrame(
        values=desc.values[dkeep],
        weights=desc.weights[dkeep],
        units=desc.units[dkeep],
    )
    _, expected = _aggregate_features(
        sub_desc, sub, order2.target_distance_weights[keep]
    )
    got = order2.at("features", "by_category", order=1).distances.xs(
        ("gender", "m"), level=("label", "cat")
    )
    pd.testing.assert_frame_equal(
        got.sort_index(), expected.sort_index(), check_names=False
    )


def test_view_order_out_of_range(order2):
    with pytest.raises(ValueError, match="order must be between 1 and split_order=2"):
        order2.at("domains", "by_attribute", order=3)


def test_combine_mismatched_split_order_raises(pops, order2):
    obs, syn, ta, sa = pops
    order1 = compare(
        obs,
        {"m": syn},
        target_attributes=ta,
        synthetic_attributes={"m": sa},
        split_on=list(ATTRS),
    )
    with pytest.raises(ValueError, match="different split_order"):
        combine({"a": order1, "b": order2})
    merged = combine({"a": order2, "b": order2})
    assert merged.split_order == 2
    assert not merged.at("domains", "by_category", order=1).distances.empty


def test_save_writes_order1_views(tmp_path, order2):
    order2.save(tmp_path)
    assert (tmp_path / "domains_by_attribute_order1" / "distances.csv").exists()
    assert (tmp_path / "features_by_category_order1" / "distances.csv").exists()


# --- CLI -----------------------------------------------------------------------


def test_cli_split_order(tmp_path, pops, capsys):
    obs, syn, ta, sa = pops
    paths = {}
    for name, df in {"obs": obs, "syn": syn, "ta": ta, "sa": sa}.items():
        paths[name] = str(tmp_path / f"{name}.csv")
        df.to_csv(paths[name], index=False)
    out = tmp_path / "out"
    args = _build_parser().parse_args(
        ["compare", paths["obs"], "-a", paths["ta"], "-m", "m", paths["syn"]]
        + [paths["sa"], "--split-on", *ATTRS, "--split-order", "2"]
        + ["--min-split-size", "0", "--no-progress", "-o", str(out)]
    )
    _run(args)
    assert "pooled over order-2 slices" in capsys.readouterr().out
    assert (out / "groups_by_category_order1" / "distances.csv").exists()


def test_cli_split_order_requires_split_on(tmp_path, pops):
    obs, syn, _, _ = pops
    obs.to_csv(tmp_path / "o.csv", index=False)
    syn.to_csv(tmp_path / "s.csv", index=False)
    args = _build_parser().parse_args(
        ["compare", str(tmp_path / "o.csv"), "-m", "m", str(tmp_path / "s.csv")]
        + ["--split-order", "2", "--no-progress"]
    )
    with pytest.raises(SystemExit, match="require --split-on"):
        _run(args)
