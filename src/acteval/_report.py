from pathlib import Path

from pandas import DataFrame


def print_markdown(data: DataFrame):
    print(data.to_markdown(tablefmt="fancy_grid", floatfmt=".3f"))


def rank(data: DataFrame) -> DataFrame:
    r = data.drop(["observed", "unit"], axis=1, errors="ignore").rank(
        axis=1, method="min"
    )
    col_ranks = r.sum(axis=0)
    ranked = [i for _, i in sorted(zip(col_ranks, col_ranks.index))]
    return r[ranked]


def _report_impl(
    frames: dict[str, DataFrame],
    prefix: str,
    head_grouper: list[str],
    log_dir: Path | None = None,
    head: int | None = None,
    verbose: bool = True,
    suffix: str = "",
    ranking: bool = False,
):
    if head is not None:
        frames[f"{prefix}descriptions_short"] = (
            frames[f"{prefix}descriptions"].groupby(head_grouper).head(head)
        )
        frames[f"{prefix}distances_short"] = (
            frames[f"{prefix}distances"].groupby(head_grouper).head(head)
        )
    else:
        frames[f"{prefix}descriptions_short"] = frames[f"{prefix}descriptions"]
        frames[f"{prefix}distances_short"] = frames[f"{prefix}distances"]

    if log_dir is not None:
        for name, frame in frames.items():
            frame.to_csv(Path(log_dir, f"{name}{suffix}.csv"))

    if verbose:
        print("\nDescriptions:")
        print_markdown(frames[f"{prefix}descriptions_short"])
        print("\nEvalutions (Distance):")
        print_markdown(frames[f"{prefix}distances_short"])

    print("\nGroup Descriptions:")
    print_markdown(frames[f"{prefix}group_descriptions"])
    print("\nGroup Evaluations (Distance):")
    print_markdown(frames[f"{prefix}group_distances"])
    if ranking:
        print("\nGroup Evaluations (Ranked):")
        print_markdown(rank(frames[f"{prefix}group_distances"]))

    print("\nDomain Descriptions:")
    print_markdown(frames[f"{prefix}domain_descriptions"])
    print("\nDomain Evaluations (Distance):")
    print_markdown(frames[f"{prefix}domain_distances"])
    if ranking:
        print("\nDomain Evaluations (Ranked):")
        print_markdown(rank(frames[f"{prefix}domain_distances"]))


def report(
    frames: dict[str, DataFrame],
    log_dir: Path | None = None,
    head: int | None = None,
    verbose: bool = True,
    suffix: str = "",
    ranking: bool = False,
):
    _report_impl(
        frames, "", ["domain", "feature"], log_dir, head, verbose, suffix, ranking
    )


def report_splits(
    frames: dict[str, DataFrame],
    log_dir: Path | None = None,
    head: int | None = None,
    verbose: bool = True,
    suffix: str = "",
    ranking: bool = False,
):
    _report_impl(
        frames,
        "label_",
        ["domain", "feature", "label"],
        log_dir,
        head,
        verbose,
        suffix,
        ranking,
    )
