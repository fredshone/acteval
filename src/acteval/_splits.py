import re

import numpy as np

from acteval.features._pid_features import PidFeatures
from acteval._jobs import JobSpec, get_jobs

_TRAILING_DIGITS = re.compile(r"\d+$")


def _get_density_jobs(config_path=None) -> list[JobSpec]:
    """Return flat list of JobSpec for all active density jobs."""
    jobs, _, _ = get_jobs(config_path)
    return jobs



def _key_activities(key: str) -> frozenset[str] | None:
    """Return activity names referenced by a feature key, or None for non-activity keys.

    Key formats:
    - n-gram transitions: "act1>act2>act3" (split by ">")
    - joint participation: "act1+act2" (split by "+")
    - timing / single-act: "actN" or "act" (strip trailing digits)
    - non-activity keys like "sequence lengths" contain spaces → return None
    """
    if ">" in key:
        return frozenset(key.split(">"))
    if "+" in key:
        return frozenset(key.split("+"))
    stripped = _TRAILING_DIGITS.sub("", key)
    if not stripped or " " in stripped:
        return None
    return frozenset([stripped])


def _subset_pid_features(
    pid_features: dict[str, dict[tuple, "PidFeatures"]],
    dense_pids: dict[str, np.ndarray],
    subset_acts: dict[str, frozenset[str]] | None = None,
) -> dict[str, dict[tuple, dict]]:
    """Subset each model's PidFeatures to dense_pids and aggregate.

    Empty entries and entries for activities absent from the subset are dropped
    so the result matches what ``feature_fn(Population(sub_df))`` would return.

    Args:
        subset_acts: Optional ``{model: frozenset_of_activity_names}`` used to
            filter out keys referencing activities not in the subset.
    """
    result: dict[str, dict[tuple, dict]] = {}
    for model, model_features in pid_features.items():
        acts = subset_acts[model] if subset_acts is not None else None
        model_result: dict[tuple, dict] = {}
        for key, pf in model_features.items():
            aggregated = pf.subset(dense_pids[model]).aggregate()
            filtered: dict = {}
            for k, v in aggregated.items():
                if len(v[0]) == 0:
                    continue
                if acts is not None:
                    key_acts = _key_activities(k)
                    if key_acts is not None and not key_acts.issubset(acts):
                        continue
                filtered[k] = v
            model_result[key] = filtered
        result[model] = model_result
    return result
