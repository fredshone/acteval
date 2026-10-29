import pytest
from pandas import DataFrame

from acteval.evaluate import compare
from acteval.features.creativity import (
    conservatism,
    diversity,
    encode_hashes,
    hash_per_pid,
    hash_population,
    hash_schedule,
    homogeneity,
    novelty,
)
from acteval.population import Population


def test_hash_schedule():
    schedule = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
        ]
    )
    assert hash_schedule(Population(schedule)) == "'home':10.0,'work':10.0,'home':10.0"


def test_hash_population_no_collisions():
    # Without separators "a1"+0.0 and "a"+10.0 would both hash to "a10.0".
    population = DataFrame(
        [
            {"pid": 0, "act": "a1", "duration": 0.0},
            {"pid": 1, "act": "a", "duration": 10.0},
        ]
    )
    assert len(hash_population(Population(population))) == 2


def test_hash_population():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 10},
        ]
    )
    assert hash_population(Population(population)) == {
        "'home':10.0,'work':10.0",
        "'home':10.0,'work':10.0,'home':10.0",
    }


def test_internal_uniqueness_full():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 10},
        ]
    )
    hashes = list(hash_per_pid(Population(population)).values())
    assert diversity(hashes) == 1
    assert homogeneity(hashes) == 0


def test_internal_uniqueness_none():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
        ]
    )
    hashes = list(hash_per_pid(Population(population)).values())
    assert diversity(hashes) == 0
    assert homogeneity(hashes) == 1


def test_diversity_is_probability_two_people_differ():
    # 4 people: a, a, b, c -> 6 pairs, 1 identical (a, a)
    assert diversity(["a", "a", "b", "c"]) == pytest.approx(5 / 6)
    assert homogeneity(["a", "a", "b", "c"]) == pytest.approx(1 / 6)


def test_diversity_shared_mode_costs_about_its_share_squared():
    # 40% share one schedule, the rest are unique: ~1 - 0.4**2, not 1 - 0.4
    hashes = ["home"] * 400 + [str(i) for i in range(600)]
    assert diversity(hashes) == pytest.approx(1 - 0.4**2, abs=1e-3)


def test_encode_hashes_shares_codes_with_vocabulary():
    target_codes, vocab = encode_hashes(["a", "b", "a"])
    assert list(target_codes) == [0, 1, 0]
    synth_codes, same_vocab = encode_hashes(["b", "c", "c", "a"], vocab)
    assert same_vocab is vocab
    # known hashes keep the target's codes; unknown ones get new codes after it
    assert list(synth_codes) == [1, 2, 2, 0]
    assert novelty(target_codes, synth_codes) == pytest.approx(1 / 3)
    assert diversity(synth_codes) == diversity(["b", "c", "c", "a"])


def test_diversity_edge_cases():
    assert diversity([]) == 0
    assert diversity(["a"]) == 1


def test_novelty_none():
    a = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 10},
        ]
    )
    b = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 10},
        ]
    )
    assert novelty(hash_population(Population(a)), hash_population(Population(b))) == 0
    assert (
        conservatism(hash_population(Population(a)), hash_population(Population(b)))
        == 1
    )


def test_novelty_full():
    a = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 10},
        ]
    )
    b = DataFrame(
        [
            {"pid": 2, "act": "home", "duration": 10},
            {"pid": 2, "act": "work", "duration": 15},
            {"pid": 2, "act": "home", "duration": 5},
            {"pid": 3, "act": "home", "duration": 10},
            {"pid": 3, "act": "work", "duration": 10},
            {"pid": 3, "act": "shop", "duration": 10},
        ]
    )
    assert novelty(hash_population(Population(a)), hash_population(Population(b))) == 1
    assert (
        conservatism(hash_population(Population(a)), hash_population(Population(b)))
        == 0
    )


def test_novelty_partial():
    a = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 10},
        ]
    )
    b = DataFrame(
        [
            {"pid": 2, "act": "home", "duration": 10},
            {"pid": 2, "act": "work", "duration": 10},
            {"pid": 2, "act": "home", "duration": 10},
            {"pid": 3, "act": "home", "duration": 10},
            {"pid": 3, "act": "work", "duration": 10},
            {"pid": 3, "act": "shop", "duration": 10},
        ]
    )
    assert (
        novelty(hash_population(Population(a)), hash_population(Population(b))) == 0.5
    )
    assert (
        conservatism(hash_population(Population(a)), hash_population(Population(b)))
        == 0.5
    )


def test_compare_end_to_end_with_only_start_end_no_duration_column():
    """Regression test: the observed contract is "any two of start/end/duration
    are sufficient" — a population with only start/end (no duration column)
    must not break creativity hashing, which used to reach for a raw
    ``.duration`` column directly."""
    observed = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 6},
            {"pid": 0, "act": "work", "start": 6, "end": 14},
            {"pid": 0, "act": "home", "start": 14, "end": 24},
            {"pid": 1, "act": "home", "start": 0, "end": 10},
            {"pid": 1, "act": "work", "start": 10, "end": 24},
        ]
    )
    synthetic = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 6},
            {"pid": 0, "act": "shop", "start": 6, "end": 14},
            {"pid": 0, "act": "home", "start": 14, "end": 24},
            {"pid": 1, "act": "home", "start": 0, "end": 12},
            {"pid": 1, "act": "work", "start": 12, "end": 24},
        ]
    )
    result = compare(observed, {"my_model": synthetic})
    assert "creativity" in result.domains.combined.distances.index
