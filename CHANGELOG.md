# Changelog

All notable changes to acteval are documented here. Entries before 0.5.0 were
reconstructed from the GitHub release notes and commit history.

## [0.5.0] - 2026-10-29

### Changed

- **Diversity is now the Gini-Simpson index** ([#14]). `creativity.diversity`
  is the probability that two different people, drawn at random, have
  different schedules (`1 - Σ nᵢ(nᵢ-1) / (N(N-1))`), replacing the ratio of
  unique schedules to people. A schedule shared by a fraction `p` of people
  now costs about `p²` rather than `p`, so a large common mode (e.g. staying at
  home all day) no longer dominates the score. `homogeneity` (its distance) is
  `1 - diversity` as before. **Diversity and homogeneity values, and so the
  creativity domain score, are not comparable with earlier releases** — on
  NTS 2023 data with ~41% stay-at-home plans, model homogeneity fell from
  0.44 to 0.18.
- Novelty and conservatism are unchanged: still counted over distinct
  schedules, as a memorisation check.
- Diversity/homogeneity units renamed from `prob. unique` / `prob. not unique`
  to `prob. pair distinct` / `prob. pair identical`.
- `creativity.diversity()` and `creativity.homogeneity()` now take one hashed
  schedule (or integer code) per person, duplicates included, instead of a
  schedule DataFrame and a set of hashes.
- `creativity.novelty()` and `creativity.conservatism()` accept arrays of
  hashes or integer codes as well as sets.

### Added

- `creativity.encode_hashes()` maps hashed schedules to integer codes that can
  be shared between populations (target vocabulary first, unseen schedules
  after it).

### Performance

- The target population is hashed and feasibility-checked once, then subset
  per slice, instead of rebuilding a `Population` from a copied DataFrame for
  every slice. Slices now hold dense pid arrays rather than DataFrame copies.
- Creativity and novel-schedule feasibility work on integer schedule codes
  with numpy, replacing per-person dict lookups and string sets.
- Together: creativity + feasibility at `split_order=2` (~460k persons,
  8 attributes) runs ~5× faster (~150s → ~28s) with half the peak memory
  (2.1 GB → 1.0 GB), with identical results.

### Internal

- Benchmarks: dropped `compare[100k]`; the split-order benchmarks run a fixed
  3 rounds without calibration, cutting the benchmark suite runtime by ~60%.
- README: describes how diversity and novelty are counted.

## [0.4.0] - 2026-10-28

### Changed

- **Fairer segment weighting** ([#13]). Each segment's aggregation weight is
  now how often it occurs (`segment_count`): summed per-person counts for
  count features (e.g. the number of `home>work` bigrams), and the number of
  values for per-occurrence features (e.g. one start time per `work0`
  episode). Rare segments, including ones only one side produces, carry
  little weight. **Participations and transitions distances are not
  comparable with 0.3.0** — they typically rise, as segments the model
  invents or misses keep their full weight.

### Added

- Warning when a `split_on` column shares no values with the target
  attributes for a model (usually a dtype mismatch such as `1` vs `"1"`),
  which would otherwise silently leave every split empty.
- Clear `ValueError` when a `split_on` column binned as numeric in the target
  attributes is not numeric in a model's attributes.

## [0.3.0] - 2026-10-27

### Added

- **N-order attribute slicing** ([#10]). `split_order` (and the `[splits]
  order` config key) scores joint assignments of several `split_on`
  attributes, e.g. `split_order=2` scores every gender & income combination.
  `result.at(level, split, order=k)` pools results back down to lower orders.
  `min_split_size` (`[splits] min_size`) skips slices with too few target
  persons; by default no minimum at order 1 and 30 above.

### Fixed

- Correctness fixes ([#11]):
  - Creativity schedule hashes no longer collide: activities are quoted and
    episodes separated, and durations are normalised to float, so integer and
    float durations of the same schedule now match. **Novelty/conservatism
    can differ from 0.2.0**, notably when target and synthetic durations had
    different dtypes (previously nothing matched, so novelty read as 1).
  - Distance aggregation weights each row by the larger of the model's and
    the target's weight (was their mean), so a segment present on only one
    side keeps that side's full weight. Model weights are rescaled to the
    target's slice size so neither side dominates through sample size.
  - Target weights stay aligned with distances when feature rows are dropped
    before group-level aggregation.
  - Segments missing from a slice are dropped rather than scored, unless the
    other side has them.
  - Schedules and attributes must now cover the same pids (raises
    `ValueError`), and the model names `target` and `unit` are reserved.

### Performance

- Per-segment EMDs run serially; the thread pool cost more than it saved.

### Internal

- Safer, same-runner benchmark comparisons in CI.

## [0.2.0] - 2026-10-08

### Added

- `compare_many()` and `compare_grid()` for multi-target comparisons, with
  results merged into one `EvalResult` by `combine()` ([#8]).
- `config_path=` on `compare()` for a custom `config.toml`; `disable=[...]`
  applies dotted-path overrides on top, and `list_disable_keys()` lists them.
- `acteval filter` CLI subcommand, and an improved `acteval compare` CLI
  ([#2]), including multiple `--split-on` arguments.
- Progress bars (`progress=True`).
- Novel-schedule feasibility checks are optional (default off).

### Changed

- Creativity and feasibility configuration refactored into their own config
  sections.
- README rewritten.

### Fixed

- Unfinished progress bars ([#7]).
- Null joins in aggregation; NaN aggregation results are filled with 0.
- `ValueError` when no evaluation jobs are enabled.
- Discovery on empty DataFrames, and a plotting colormap bug.

## [0.1.2] - 2026-04-30

### Added

- `acteval` command-line interface ([#1]). Accepts any two of
  `start`/`end`/`duration` and derives the third.
- `llms.txt` agent guide.

### Changed

- Requires Python 3.10 or later.

## [0.1.1] - 2026-04-27

First tagged release.

- `Population`, `Evaluator` and `compare()` for evaluating synthetic activity
  schedules against observed data across participations, transitions,
  timing, creativity and feasibility domains.
- `EvalResult` with feature / group / domain aggregation levels and
  combined / by-attribute / by-category views.
- Optional attribute splits, including binned numeric attributes.
- pandas and polars input.
- `pairwise_distances()` with pluggable distance specs.
- Matplotlib plotting helpers and documentation.

[#1]: https://github.com/fredshone/acteval/pull/1
[#2]: https://github.com/fredshone/acteval/pull/2
[#7]: https://github.com/fredshone/acteval/pull/7
[#8]: https://github.com/fredshone/acteval/pull/8
[#10]: https://github.com/fredshone/acteval/pull/10
[#11]: https://github.com/fredshone/acteval/pull/11
[#13]: https://github.com/fredshone/acteval/pull/13
[#14]: https://github.com/fredshone/acteval/pull/14
[0.5.0]: https://github.com/fredshone/acteval/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/fredshone/acteval/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/fredshone/acteval/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/fredshone/acteval/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/fredshone/acteval/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/fredshone/acteval/releases/tag/v0.1.1
