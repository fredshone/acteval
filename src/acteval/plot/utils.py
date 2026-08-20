import numpy as np
import pandas as pd

from acteval.population import Population


def _to_population(data) -> Population:
    """Accept a Population or DataFrame and return a Population."""
    if isinstance(data, Population):
        return data
    return Population(data)


class PopulationGenerator:
    """Generate synthetic activity schedule populations with configurable profile biases.

    Seven profiles cover the full activity vocabulary; ``profile_weights`` controls
    how likely each one is relative to the others.  All profiles are always reachable
    (any profile with a missing or zero weight gets a small floor weight of 0.3),
    so every population retains diversity while still expressing a clear mode.

    Available profiles
    ------------------
    office         — conventional 9-to-5 with an eat_out lunch break
    flexible       — variable-hours work followed by shopping
    student        — morning education then leisure
    part_time      — short mid-morning work block
    leisure_shopper — leisure activity then shopping, no work
    late_worker    — spread or late work hours
    home_heavy     — extended leisure, mostly at home
    """

    _PROFILES = [
        "office",
        "flexible",
        "student",
        "part_time",
        "leisure_shopper",
        "late_worker",
        "home_heavy",
    ]
    _FLOOR_WEIGHT = 0.3

    def __init__(self, profile_weights: dict[str, float] | None = None, seed: int = 42):
        raw = {p: self._FLOOR_WEIGHT for p in self._PROFILES}
        for p, w in (profile_weights or {}).items():
            raw[p] = max(w, self._FLOOR_WEIGHT)
        total = sum(raw.values())
        self._probs = [raw[p] / total for p in self._PROFILES]
        self._seed = seed

    def __call__(self, n: int = 500) -> pd.DataFrame:
        return self.generate(n)

    def generate(self, n: int = 500) -> pd.DataFrame:
        rng = np.random.default_rng(self._seed)
        rows = []
        for pid in range(n):
            profile = rng.choice(self._PROFILES, p=self._probs)
            for act, start, end in self._episodes(rng, profile):
                rows.append(
                    {
                        "pid": pid,
                        "act": act,
                        "start": start,
                        "end": end,
                        "duration": end - start,
                    }
                )
        return pd.DataFrame(rows)

    def _episodes(self, rng, profile: str) -> list[tuple[str, int, int]]:
        if profile == "office":
            work_start = _clip_normal(rng, 450, 30, 390, 510)
            lunch_start = _clip_normal(rng, 810, 15, max(work_start + 60, 750), 870)
            lunch_end = lunch_start + 60
            work_end = _clip_normal(rng, 1080, 20, max(lunch_end + 60, 1020), 1140)
            return [
                ("home", 0, work_start),
                ("work", work_start, lunch_start),
                ("eat_out", lunch_start, lunch_end),
                ("work", lunch_end, work_end),
                ("home", work_end, 1440),
            ]
        if profile == "flexible":
            work_start = _clip_normal(rng, 480, 45, 360, 600)
            work_end = _clip_normal(rng, 960, 30, max(work_start + 60, 900), 1080)
            shop_end = min(work_end + 60, 1440)
            return [
                ("home", 0, work_start),
                ("work", work_start, work_end),
                ("shop", work_end, shop_end),
                ("home", shop_end, 1440),
            ]
        if profile == "student":
            edu_start = _clip_normal(rng, 540, 30, 480, 600)
            edu_end = _clip_normal(rng, 840, 20, max(edu_start + 60, 780), 900)
            lei_end = min(edu_end + 120, 1440)
            return [
                ("home", 0, edu_start),
                ("education", edu_start, edu_end),
                ("leisure", edu_end, lei_end),
                ("home", lei_end, 1440),
            ]
        if profile == "part_time":
            work_start = _clip_normal(rng, 540, 30, 480, 600)
            work_end = _clip_normal(rng, 900, 30, max(work_start + 60, 840), 960)
            return [
                ("home", 0, work_start),
                ("work", work_start, work_end),
                ("home", work_end, 1440),
            ]
        if profile == "leisure_shopper":
            lei_start = _clip_normal(rng, 420, 60, 300, 540)
            lei_end = _clip_normal(rng, 780, 60, max(lei_start + 120, 600), 900)
            shop_end = min(lei_end + 90, 1440)
            return [
                ("home", 0, lei_start),
                ("leisure", lei_start, lei_end),
                ("shop", lei_end, shop_end),
                ("home", shop_end, 1440),
            ]
        if profile == "late_worker":
            work_start = _clip_normal(rng, 600, 90, 360, 900)
            work_end = _clip_normal(rng, 1080, 60, max(work_start + 120, 960), 1380)
            return [
                ("home", 0, work_start),
                ("work", work_start, work_end),
                ("home", work_end, 1440),
            ]
        # home_heavy
        lei_start = _clip_normal(rng, 600, 60, 480, 720)
        lei_end = _clip_normal(rng, 1200, 60, max(lei_start + 120, 1080), 1380)
        return [
            ("home", 0, lei_start),
            ("leisure", lei_start, lei_end),
            ("home", lei_end, 1440),
        ]


def _clip_normal(rng, mean, std, lo, hi):
    return int(round(np.clip(rng.normal(mean, std), lo, hi)))
