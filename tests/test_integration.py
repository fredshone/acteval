"""Integration tests covering the full evaluation pipeline schema.

These tests run the pipeline end-to-end and assert on DataFrame schemas at each
aggregation level. They serve as the safety net for refactoring _aggregation.py,
_pipeline.py, and evaluate.py.

The `observed` and `synthetic` fixtures come from conftest.py.
"""
import pytest
from pandas import DataFrame

from acteval.evaluate import Evaluator, compare, compare_splits


# ---------------------------------------------------------------------------
# Extra fixtures for split-based tests
# ---------------------------------------------------------------------------


@pytest.fixture
def obs_attrs():
    return DataFrame({"pid": [0, 1], "group": ["a", "b"]})


@pytest.fixture
def synth_attrs():
    return DataFrame({"pid": [0, 1], "group": ["a", "b"]})


# ---------------------------------------------------------------------------
# Schema tests — no splits
# ---------------------------------------------------------------------------


class TestNoSplitsSchema:
    def test_all_keys_present(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert set(result.keys()) == {
            "descriptions",
            "distances",
            "group_descriptions",
            "group_distances",
            "domain_descriptions",
            "domain_distances",
        }

    def test_no_label_keys_without_splits(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert not any(k.startswith("label_") for k in result.keys())

    def test_distances_index_names(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert result.distances.index.names == ["domain", "feature", "segment"]

    def test_group_distances_index_names(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert result.group_distances.index.names == ["domain", "feature"]

    def test_domain_distances_index_names(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert result.domain_distances.index.names == ["domain"]

    def test_distances_has_model_column(self, observed, synthetic):
        # distances has model columns + unit; observed__weight is the weighting
        # reference but "observed" is not a distance value column
        result = compare(observed, {"m": synthetic})
        assert "m" in result.distances.columns

    def test_descriptions_has_observed_column(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert "observed" in result.descriptions.columns
        assert "m" in result.descriptions.columns

    def test_distances_has_no_weight_columns_after_aggregation(self, observed, synthetic):
        # Weight columns are consumed in _aggregate_features; only values + unit remain.
        result = compare(observed, {"m": synthetic})
        assert not any(c.endswith("__weight") for c in result.distances.columns)

    def test_group_distances_has_no_weight_columns(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert not any(c.endswith("__weight") for c in result.group_distances.columns)

    def test_domain_distances_has_no_weight_or_unit_columns(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        cols = set(result.domain_distances.columns)
        assert not any(c.endswith("__weight") for c in cols)
        assert "unit" not in cols

    def test_domain_distances_values_in_range(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        vals = result.domain_distances["m"]
        assert (vals >= 0).all(), f"Negative distances: {vals}"
        assert (vals <= 1).all(), f"Distances > 1: {vals}"

    def test_group_distances_values_in_range(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        vals = result.group_distances["m"]
        assert (vals >= 0).all()
        assert (vals <= 1).all()

    def test_descriptions_has_unit_column(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert "unit" in result.descriptions.columns

    def test_group_descriptions_has_unit_column(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert "unit" in result.group_descriptions.columns

    def test_domain_descriptions_has_no_unit_column(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert "unit" not in result.domain_descriptions.columns

    def test_descriptions_index_names(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert result.descriptions.index.names == ["domain", "feature", "segment"]

    def test_group_descriptions_index_names(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert result.group_descriptions.index.names == ["domain", "feature"]

    def test_domain_descriptions_index_names(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert result.domain_descriptions.index.names == ["domain"]

    def test_self_comparison_distances_in_range(self, observed):
        result = compare(observed, {"m": observed})
        vals = result.domain_distances["m"]
        assert (vals >= 0).all()
        assert (vals <= 1).all()

    def test_multiple_models(self, observed, synthetic):
        result = compare(
            observed, {"m1": synthetic, "m2": synthetic}
        )
        assert "m1" in result.domain_distances.columns
        assert "m2" in result.domain_distances.columns

    def test_model_names_property(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert result.model_names == ["m"]

    def test_summary_returns_model_columns_only(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        summary = result.summary()
        assert "observed" not in summary.columns
        assert "m" in summary.columns


# ---------------------------------------------------------------------------
# Schema tests — with splits
# ---------------------------------------------------------------------------


class TestWithSplitsSchema:
    def test_label_keys_present_with_splits(
        self, observed, synthetic, obs_attrs, synth_attrs
    ):
        evaluator = Evaluator(
            observed, target_attributes=obs_attrs, split_on=["group"]
        )
        result = evaluator.compare(
            {"m": synthetic},
            attributes={"m": synth_attrs},

        )
        for key in (
            "label_descriptions",
            "label_distances",
            "label_group_descriptions",
            "label_group_distances",
            "label_domain_descriptions",
            "label_domain_distances",
        ):
            assert key in result.keys(), f"Missing key: {key}"

    def test_label_distances_has_weight_columns(
        self, observed, synthetic, obs_attrs, synth_attrs
    ):
        """Raw label_distances preserves observed__weight and model__weight."""
        evaluator = Evaluator(
            observed, target_attributes=obs_attrs, split_on=["group"]
        )
        result = evaluator.compare(
            {"m": synthetic},
            attributes={"m": synth_attrs},

        )
        assert "observed__weight" in result.label_distances.columns
        assert "m__weight" in result.label_distances.columns

    def test_label_distances_five_level_index(
        self, observed, synthetic, obs_attrs, synth_attrs
    ):
        evaluator = Evaluator(
            observed, target_attributes=obs_attrs, split_on=["group"]
        )
        result = evaluator.compare(
            {"m": synthetic},
            attributes={"m": synth_attrs},

        )
        assert result.label_distances.index.names == [
            "domain",
            "feature",
            "segment",
            "label",
            "cat",
        ]

    def test_label_distances_has_expected_categories(
        self, observed, synthetic, obs_attrs, synth_attrs
    ):
        evaluator = Evaluator(
            observed, target_attributes=obs_attrs, split_on=["group"]
        )
        result = evaluator.compare(
            {"m": synthetic},
            attributes={"m": synth_attrs},

        )
        cats = set(result.label_distances.index.get_level_values("cat").unique())
        assert cats == {"a", "b"}

    def test_label_domain_distances_has_three_level_index(
        self, observed, synthetic, obs_attrs, synth_attrs
    ):
        evaluator = Evaluator(
            observed, target_attributes=obs_attrs, split_on=["group"]
        )
        result = evaluator.compare(
            {"m": synthetic},
            attributes={"m": synth_attrs},

        )
        assert result.label_domain_distances.index.names == ["domain", "label"]

    def test_distances_collapses_over_splits(
        self, observed, synthetic, obs_attrs, synth_attrs
    ):
        """Top-level distances should have the 3-level index (no label/cat)."""
        evaluator = Evaluator(
            observed, target_attributes=obs_attrs, split_on=["group"]
        )
        result = evaluator.compare(
            {"m": synthetic},
            attributes={"m": synth_attrs},

        )
        assert result.distances.index.names == ["domain", "feature", "segment"]

    def test_split_domain_distances_in_range(
        self, observed, synthetic, obs_attrs, synth_attrs
    ):
        evaluator = Evaluator(
            observed, target_attributes=obs_attrs, split_on=["group"]
        )
        result = evaluator.compare(
            {"m": synthetic},
            attributes={"m": synth_attrs},

        )
        vals = result.label_domain_distances["m"]
        assert (vals >= 0).all()
        assert (vals <= 1).all()
