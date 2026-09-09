from numpy import array
from pandas import DataFrame, MultiIndex, Series

from acteval._aggregation import (
    descriptions_to_domain_level,
    descriptions_to_group_level,
    distances_to_domain_level,
    distances_to_group_level,
)
from acteval._pipeline import _aggregate_features
from acteval._result_frame import ResultFrame
from acteval.features._utils import equals
from acteval.features.structural import (
    feasibility_eval,
    time_consistency,
)
from acteval.population import Population


def _describe(
    descriptions: ResultFrame, distances: ResultFrame, target_weights: Series
):
    """Three-tier aggregation, mirroring what evaluate.py assembles internally."""
    feat_desc, feat_dist = _aggregate_features(descriptions, distances, target_weights)
    group_desc = descriptions_to_group_level(descriptions)
    group_dist = distances_to_group_level(distances, target_weights)
    domain_desc = descriptions_to_domain_level(group_desc)
    domain_dist = distances_to_domain_level(group_dist)
    return {
        "descriptions": feat_desc,
        "distances": feat_dist,
        "group_descriptions": group_desc.values,
        "group_distances": group_dist.values,
        "domain_descriptions": domain_desc.values,
        "domain_distances": domain_dist.values,
    }


def test_time_consistency():
    population = DataFrame(
        [
            {"pid": 0, "start": 0, "end": 10, "duration": 10},
            {"pid": 0, "start": 10, "end": 20, "duration": 10},
            {"pid": 0, "start": 20, "end": 30, "duration": 10},
            {"pid": 1, "start": 0, "end": 10, "duration": 10},
            {"pid": 1, "start": 10, "end": 20, "duration": 10},
        ]
    )
    result = time_consistency(Population(population), target=30).aggregate()
    assert set(result.keys()) == {"starts at 0", "ends at 30", "duration is 30"}
    # Both persons start at 0 → only value 1, count 2
    assert equals(
        result,
        {
            "starts at 0": (array([1.0]), array([2])),
            "ends at 30": (array([0.0, 1.0]), array([1, 1])),
            "duration is 30": (array([0.0, 1.0]), array([1, 1])),
        },
    )


def test_feasibility_eval():
    schedule = DataFrame(
        [
            {"pid": 0, "act": "home"},
            {"pid": 0, "act": "work"},
            {"pid": 0, "act": "home"},
            {"pid": 1, "act": "home"},
            {"pid": 1, "act": "home"},
            {"pid": 2, "act": "home"},
            {"pid": 2, "act": "work"},
            {"pid": 2, "act": "shop"},
        ]
    )
    weights, metrics = feasibility_eval(Population(schedule), "observed")
    assert (
        weights.reset_index(drop=True)
        .astype("int32")
        .equals(Series([3, 3, 3, 3, 3, 3, 3, 3], dtype="int32"))
    )
    assert metrics.reset_index(drop=True).equals(
        Series([2 / 3, 1 / 3, 0, 1 / 3, 1 / 3, 1 / 3, 0, 0])
    )


def test_feasibility_eval_out_of_order_rows():
    # Person 0's rows are in the DataFrame as: home@0, home@960, work@480.
    # The true time-ordered sequence is home→work→home — no consecutive home.
    # Without the (pid, start) sort fix this was spuriously flagged.
    schedule = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 480, "duration": 480},
            {"pid": 0, "act": "home", "start": 960, "end": 1440, "duration": 480},
            {"pid": 0, "act": "work", "start": 480, "end": 960, "duration": 480},
        ]
    )
    _, metrics = feasibility_eval(Population(schedule), "test")
    assert metrics.reset_index(drop=True).equals(
        Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )


def test_describe_structural():
    index = MultiIndex.from_tuples(
        [
            ("feasibility", "invalid", "all"),
            ("feasibility", "not home based", "all"),
            ("feasibility", "not home based", "starts"),
            ("feasibility", "not home based", "ends"),
            ("feasibility", "consecutive", "all"),
            ("feasibility", "consecutive", "home"),
            ("feasibility", "consecutive", "work"),
            ("feasibility", "consecutive", "education"),
        ],
        names=["domain", "feature", "segment"],
    )

    weight = Series([3, 3, 3, 3, 3, 3, 3, 3], index=index)
    value = Series([2 / 3, 1 / 3, 0, 1 / 3, 1 / 3, 1 / 3, 0, 0], index=index)
    unit = Series(["prob. invalid"] * 8, index=index)

    descriptions = ResultFrame(
        values=DataFrame({"target": value, "y": value}),
        weights=DataFrame({"target": weight, "y": weight}),
        units=unit,
    )
    distances = ResultFrame(
        values=DataFrame({"y": value}),
        weights=DataFrame({"y": weight}),
        units=unit,
    )
    frames = _describe(descriptions, distances, target_weights=weight)
    assert len(frames["descriptions"]) == 8
    assert len(frames["group_descriptions"]) == 3
    assert len(frames["domain_descriptions"]) == 1

    assert len(frames["distances"]) == 8
    assert len(frames["group_distances"]) == 3
    assert len(frames["domain_distances"]) == 1


def test_describe_splits_structural():
    index = MultiIndex.from_tuples(
        [
            ("feasibility", "invalid", "all", "a"),
            ("feasibility", "not home based", "all", "a"),
            ("feasibility", "not home based", "starts", "a"),
            ("feasibility", "not home based", "ends", "a"),
            ("feasibility", "consecutive", "all", "a"),
            ("feasibility", "consecutive", "home", "a"),
            ("feasibility", "consecutive", "work", "a"),
            ("feasibility", "consecutive", "education", "a"),
            ("feasibility", "invalid", "all", "b"),
            ("feasibility", "not home based", "all", "b"),
            ("feasibility", "not home based", "starts", "b"),
            ("feasibility", "not home based", "ends", "b"),
            ("feasibility", "consecutive", "all", "b"),
            ("feasibility", "consecutive", "home", "b"),
            ("feasibility", "consecutive", "work", "b"),
            ("feasibility", "consecutive", "education", "b"),
        ],
        names=["domain", "feature", "segment", "label"],
    )

    weight = Series([3] * 16, index=index)
    value = Series([1 / 3, 0] * 8, index=index)
    unit = Series(["prob. invalid"] * 16, index=index)

    descriptions = ResultFrame(
        values=DataFrame({"target": value, "y": value}),
        weights=DataFrame({"target": weight, "y": weight}),
        units=unit,
    )
    distances = ResultFrame(
        values=DataFrame({"y": value}),
        weights=DataFrame({"y": weight}),
        units=unit,
    )
    frames = _describe(descriptions, distances, target_weights=weight)
    assert len(frames["descriptions"]) == 8
    assert len(frames["group_descriptions"]) == 3
    assert len(frames["domain_descriptions"]) == 1

    assert len(frames["distances"]) == 8
    assert len(frames["group_distances"]) == 3
    assert len(frames["domain_distances"]) == 1

    label_group_desc = descriptions_to_group_level(descriptions, extra=["label"])
    label_group_dist = distances_to_group_level(distances, weight, extra=["label"])
    label_domain_desc = descriptions_to_domain_level(label_group_desc, extra=["label"])
    label_domain_dist = distances_to_domain_level(label_group_dist, extra=["label"])
    assert len(descriptions.values) == 16
    assert len(label_group_desc.values) == 6
    assert len(label_domain_desc.values) == 2

    assert len(distances.values) == 16
    assert len(label_group_dist.values) == 6
    assert len(label_domain_dist.values) == 2
