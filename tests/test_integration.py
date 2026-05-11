"""Integration tests covering the full evaluation pipeline schema.

These tests run the pipeline end-to-end and assert on DataFrame schemas at each
aggregation level. They serve as the safety net for refactoring _aggregation.py,
_pipeline.py, and evaluate.py.

The `observed` and `synthetic` fixtures come from conftest.py.
"""

import pytest
from pandas import DataFrame

from acteval._jobs import CreativityConfig, EvalConfig, StructuralConfig, get_jobs
from acteval.evaluate import Evaluator, SplitNotAvailableError, compare

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
    def test_has_splits_false(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert result.has_splits is False

    def test_features_combined_index_names(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert result.features.combined.distances.index.names == [
            "domain",
            "feature",
            "segment",
        ]

    def test_groups_combined_index_names(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert result.groups.combined.distances.index.names == ["domain", "feature"]

    def test_domains_combined_index_names(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert result.domains.combined.distances.index.names == ["domain"]

    def test_features_combined_has_model_column(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert "m" in result.features.combined.distances.columns

    def test_features_combined_descriptions_has_observed(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert "observed" in result.features.combined.descriptions.columns
        assert "m" in result.features.combined.descriptions.columns

    def test_no_weight_columns_in_distances(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert not any(
            c.endswith("__weight") for c in result.features.combined.distances.columns
        )
        assert not any(
            c.endswith("__weight") for c in result.groups.combined.distances.columns
        )

    def test_domains_no_weight_or_unit_columns(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        cols = set(result.domains.combined.distances.columns)
        assert not any(c.endswith("__weight") for c in cols)
        assert "unit" not in cols

    def test_domains_distances_values_in_range(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        vals = result.domains.combined.distances["m"]
        assert (vals >= 0).all(), f"Negative distances: {vals}"
        assert (vals <= 1).all(), f"Distances > 1: {vals}"

    def test_groups_distances_values_in_range(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        vals = result.groups.combined.distances["m"]
        assert (vals >= 0).all()
        assert (vals <= 1).all()

    def test_features_combined_has_unit_column(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert "unit" in result.features.combined.descriptions.columns

    def test_groups_combined_has_unit_column(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert "unit" in result.groups.combined.descriptions.columns

    def test_domains_combined_no_unit_column(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert "unit" not in result.domains.combined.descriptions.columns

    def test_features_descriptions_index_names(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert result.features.combined.descriptions.index.names == [
            "domain",
            "feature",
            "segment",
        ]

    def test_groups_descriptions_index_names(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert result.groups.combined.descriptions.index.names == ["domain", "feature"]

    def test_domains_descriptions_index_names(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert result.domains.combined.descriptions.index.names == ["domain"]

    def test_self_comparison_distances_in_range(self, observed):
        result = compare(observed, {"m": observed})
        vals = result.domains.combined.distances["m"]
        assert (vals >= 0).all()
        assert (vals <= 1).all()

    def test_multiple_models(self, observed, synthetic):
        result = compare(observed, {"m1": synthetic, "m2": synthetic})
        assert "m1" in result.domains.combined.distances.columns
        assert "m2" in result.domains.combined.distances.columns

    def test_model_names_property(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        assert result.model_names == ["m"]

    def test_summary_returns_model_columns_only(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        summary = result.summary()
        assert "observed" not in summary.columns
        assert "m" in summary.columns

    def test_split_access_raises_without_splits(self, observed, synthetic):
        result = compare(observed, {"m": synthetic})
        with pytest.raises(SplitNotAvailableError):
            _ = result.domains.by_attribute
        with pytest.raises(SplitNotAvailableError):
            _ = result.features.by_category


# ---------------------------------------------------------------------------
# Schema tests — with splits
# ---------------------------------------------------------------------------


class TestWithSplitsSchema:
    def test_has_splits_true(self, observed, synthetic, obs_attrs, synth_attrs):
        evaluator = Evaluator(observed, target_attributes=obs_attrs, split_on=["group"])
        result = evaluator.compare({"m": synthetic}, attributes={"m": synth_attrs})
        assert result.has_splits is True

    def test_features_by_category_five_level_index(
        self, observed, synthetic, obs_attrs, synth_attrs
    ):
        evaluator = Evaluator(observed, target_attributes=obs_attrs, split_on=["group"])
        result = evaluator.compare({"m": synthetic}, attributes={"m": synth_attrs})
        assert result.features.by_category.distances.index.names == [
            "domain",
            "feature",
            "segment",
            "label",
            "cat",
        ]

    def test_features_by_category_has_expected_categories(
        self, observed, synthetic, obs_attrs, synth_attrs
    ):
        evaluator = Evaluator(observed, target_attributes=obs_attrs, split_on=["group"])
        result = evaluator.compare({"m": synthetic}, attributes={"m": synth_attrs})
        cats = set(
            result.features.by_category.distances.index.get_level_values("cat").unique()
        )
        assert cats == {"a", "b"}

    def test_domains_by_attribute_two_level_index(
        self, observed, synthetic, obs_attrs, synth_attrs
    ):
        evaluator = Evaluator(observed, target_attributes=obs_attrs, split_on=["group"])
        result = evaluator.compare({"m": synthetic}, attributes={"m": synth_attrs})
        assert result.domains.by_attribute.distances.index.names == ["domain", "label"]

    def test_domains_by_category_three_level_index(
        self, observed, synthetic, obs_attrs, synth_attrs
    ):
        evaluator = Evaluator(observed, target_attributes=obs_attrs, split_on=["group"])
        result = evaluator.compare({"m": synthetic}, attributes={"m": synth_attrs})
        assert result.domains.by_category.distances.index.names == [
            "domain",
            "label",
            "cat",
        ]

    def test_features_by_attribute_four_level_index(
        self, observed, synthetic, obs_attrs, synth_attrs
    ):
        evaluator = Evaluator(observed, target_attributes=obs_attrs, split_on=["group"])
        result = evaluator.compare({"m": synthetic}, attributes={"m": synth_attrs})
        assert result.features.by_attribute.distances.index.names == [
            "domain",
            "feature",
            "segment",
            "label",
        ]

    def test_combined_collapses_over_splits(
        self, observed, synthetic, obs_attrs, synth_attrs
    ):
        """Combined distances should have no label/cat index levels."""
        evaluator = Evaluator(observed, target_attributes=obs_attrs, split_on=["group"])
        result = evaluator.compare({"m": synthetic}, attributes={"m": synth_attrs})
        assert result.features.combined.distances.index.names == [
            "domain",
            "feature",
            "segment",
        ]

    def test_split_domain_distances_in_range(
        self, observed, synthetic, obs_attrs, synth_attrs
    ):
        evaluator = Evaluator(observed, target_attributes=obs_attrs, split_on=["group"])
        result = evaluator.compare({"m": synthetic}, attributes={"m": synth_attrs})
        vals = result.domains.by_attribute.distances["m"]
        assert (vals >= 0).all()
        assert (vals <= 1).all()


# ---------------------------------------------------------------------------
# Config component tests
# ---------------------------------------------------------------------------


def _minimal_jobs(creativity=None, structural=None) -> EvalConfig:
    """Build an EvalConfig with default density and supplied creativity/structural."""
    base = get_jobs()
    return EvalConfig(
        density=base.density,
        creativity=creativity if creativity is not None else base.creativity,
        structural=structural if structural is not None else base.structural,
    )


class TestConfigComponents:
    def test_creativity_diversity_only_rows(self, observed, synthetic):
        """Disabling novelty removes novelty/conservatism rows."""
        jobs = _minimal_jobs(creativity=CreativityConfig(diversity=True, novelty=False))
        result = Evaluator(observed, jobs=jobs).compare({"m": synthetic})
        features = set(result.features.combined.distances.index.get_level_values("feature"))
        assert "homogeneity" in features
        assert "novelty" not in features
        assert "conservatism" not in features

    def test_creativity_novelty_only_rows(self, observed, synthetic):
        """Disabling diversity removes diversity/homogeneity rows."""
        jobs = _minimal_jobs(creativity=CreativityConfig(diversity=False, novelty=True))
        result = Evaluator(observed, jobs=jobs).compare({"m": synthetic})
        features = set(result.features.combined.distances.index.get_level_values("feature"))
        assert "conservatism" in features
        assert "diversity" not in features
        assert "homogeneity" not in features

    def test_structural_home_based_only_rows(self, observed, synthetic):
        """Disabling consecutive removes consecutive rows; keeps not-home-based rows."""
        jobs = _minimal_jobs(
            structural=StructuralConfig(home_based=True, consecutive=False)
        )
        result = Evaluator(observed, jobs=jobs).compare({"m": synthetic})
        features = set(result.features.combined.distances.index.get_level_values("feature"))
        assert any("not home based" in f for f in features)
        assert not any("consecutive" in f for f in features)

    def test_structural_novel_job_adds_novel_rows(self, observed, synthetic):
        """Enabling home_based_novel adds '(novel)' rows to output."""
        jobs = _minimal_jobs(
            structural=StructuralConfig(home_based=False, consecutive=False, home_based_novel=True)
        )
        result = Evaluator(observed, jobs=jobs).compare({"m": synthetic})
        features = set(result.features.combined.distances.index.get_level_values("feature"))
        assert any("(novel)" in f for f in features)

    def test_novel_structural_without_creativity_does_not_error(self, observed, synthetic):
        """Novel structural jobs work even when creativity is disabled (obs_hashes still computed)."""
        jobs = _minimal_jobs(
            creativity=CreativityConfig(diversity=False, novelty=False),
            structural=StructuralConfig(
                home_based=False, consecutive=False,
                home_based_novel=True, consecutive_novel=True,
            ),
        )
        result = Evaluator(observed, jobs=jobs).compare({"m": synthetic})
        features = set(result.features.combined.distances.index.get_level_values("feature"))
        assert any("(novel)" in f for f in features)
