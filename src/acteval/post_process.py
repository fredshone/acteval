"""Post-processing functions for aggregating feature-level evaluation results.

Each function takes a feature-level DataFrame (or Series for a single model)
and returns it aggregated to the requested level. The optional ``extra``
parameter appends additional index levels to the grouper — use ``extra=["label"]``
for split-stratified aggregation.

- ``descriptions_to_group_level`` / ``distances_to_group_level``: collapse
  ``(domain, feature, segment, ...)`` → ``(domain, feature, ...)``
- ``descriptions_to_domain_level`` / ``distances_to_domain_level``: collapse
  ``(domain, feature, ...)`` → ``(domain, ...)``
"""
from pandas import DataFrame, Series

from acteval._pipeline import (
    _REMOVE_FEATURES,
    _REMOVE_GROUPS,
    _drop_features,
    distance_weighted_average,
    weighted_average,
)


def descriptions_to_group_level(
    descriptions: DataFrame | Series, extra: list[str] = []
) -> DataFrame | Series:
    """Aggregate feature-level descriptions to group level (domain, feature)."""
    grouper = ["domain", "feature"] + extra
    desc = _drop_features(descriptions.drop("unit", axis=1), _REMOVE_FEATURES)
    group_desc = desc.groupby(grouper).apply(weighted_average)
    group_desc["unit"] = descriptions["unit"].groupby(grouper).first()
    return group_desc


def distances_to_group_level(
    distances: DataFrame | Series, extra: list[str] = []
) -> DataFrame | Series:
    """Aggregate feature-level distances to group level (domain, feature)."""
    grouper = ["domain", "feature"] + extra
    dist = _drop_features(distances.drop("unit", axis=1), _REMOVE_FEATURES)
    group_dist = dist.groupby(grouper).apply(distance_weighted_average)
    group_dist["unit"] = distances["unit"].groupby(grouper).first()
    return group_dist


def descriptions_to_domain_level(
    group_descriptions: DataFrame | Series, extra: list[str] = []
) -> DataFrame | Series:
    """Aggregate group-level descriptions to domain level."""
    grouper = ["domain"] + extra
    domain_desc = _drop_features(group_descriptions.drop("unit", axis=1), _REMOVE_GROUPS)
    return domain_desc.groupby(grouper).mean()


def distances_to_domain_level(
    group_distances: DataFrame | Series, extra: list[str] = []
) -> DataFrame | Series:
    """Aggregate group-level distances to domain level."""
    grouper = ["domain"] + extra
    domain_dist = _drop_features(group_distances.drop("unit", axis=1), _REMOVE_GROUPS)
    return domain_dist.groupby(grouper).mean()
