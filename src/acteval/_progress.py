"""tqdm progress-bar helpers for Evaluator's optional ``progress=True`` output."""

from tqdm import tqdm

_ITEM_WIDTH = 25


def make_bar(
    desc: str,
    total: int,
    position: int | None = None,
    desc_width: int | None = None,
    colour: str | None = "green",
) -> tqdm:
    label = f"{desc:<{desc_width}}" if desc_width else desc
    full_desc = f"{label}  {'':>{_ITEM_WIDTH}}"
    bar_format = (
        "{desc} {percentage:3.0f}% │{bar:25}│ {n_fmt:>4}/{total_fmt} [{elapsed}]"
    )
    kwargs: dict = dict(
        total=total,
        desc=full_desc,
        leave=True,
        bar_format=bar_format,
        ascii=" ━",
        colour=colour,
    )
    if position is not None:
        kwargs["position"] = position
    bar = tqdm(**kwargs)
    bar._acteval_label = label
    return bar


def bar_set_item(bar: tqdm, item: str) -> None:
    field = f"{item:<{_ITEM_WIDTH}}"[:_ITEM_WIDTH]
    bar.set_description_str(f"{bar._acteval_label}  {field}", refresh=True)


def bar_clear_item(bar: tqdm) -> None:
    bar.set_description_str(f"{bar._acteval_label}  {'':>{_ITEM_WIDTH}}", refresh=False)
