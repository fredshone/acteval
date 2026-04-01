import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from acteval import list_features
from acteval.features.catalogue import CATALOGUE, FeatureEntry
from acteval.features._pid_features import PidFeatures
from acteval.population import Population


def _pop():
    df = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 6, "duration": 6},
            {"pid": 0, "act": "work", "start": 6, "end": 14, "duration": 8},
            {"pid": 0, "act": "home", "start": 14, "end": 24, "duration": 10},
            {"pid": 1, "act": "home", "start": 0, "end": 10, "duration": 10},
            {"pid": 1, "act": "work", "start": 10, "end": 20, "duration": 10},
            {"pid": 1, "act": "home", "start": 20, "end": 24, "duration": 4},
        ]
    )
    return Population(df)


def test_list_features_returns_dataframe():
    df = list_features()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"domain", "name", "config_key", "description", "in_default_config"}
    assert len(df) == len(CATALOGUE)


def test_catalogue_entries_are_feature_entries():
    for entry in CATALOGUE:
        assert isinstance(entry, FeatureEntry)
        assert isinstance(entry.domain, str)
        assert isinstance(entry.name, str)
        assert isinstance(entry.config_key, str)
        assert isinstance(entry.description, str)
        assert isinstance(entry.in_default_config, bool)


def test_catalogue_functions_callable():
    for entry in CATALOGUE:
        assert callable(entry.function), f"{entry.name}: function is not callable"


def test_catalogue_functions_return_pid_features():
    pop = _pop()
    for entry in CATALOGUE:
        result = entry.function(pop)
        assert isinstance(result, PidFeatures), (
            f"{entry.name}: expected PidFeatures, got {type(result)}"
        )


def test_default_config_entries_match_expected():
    default_keys = {e.config_key for e in CATALOGUE if e.in_default_config}
    expected = {"lengths", "rates", "pair_rates", "start_times", "durations",
                "start_durations", "joint_durations", "2-gram", "3-gram", "4-gram"}
    assert default_keys == expected
