"""tqdm progress-bar helpers for Evaluator's optional ``progress=True`` output."""

from collections.abc import Iterator
from contextlib import contextmanager

from tqdm import tqdm

_ITEM_WIDTH = 25


def make_bar(
    desc: str,
    total: int,
    position: int | None = None,
    desc_width: int | None = None,
    colour: str | None = "green",
    disable: bool = False,
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
        disable=disable,
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
    # refresh=True: this is called right after a bar's loop finishes, so it's
    # also the only guaranteed redraw showing the bar at 100% before it may
    # sit untouched (and thus visually frozen) until a much later close().
    bar.set_description_str(f"{bar._acteval_label}  {'':>{_ITEM_WIDTH}}", refresh=True)


@contextmanager
def bar_scope(
    bar: tqdm | None,
    desc: str,
    total: int,
    enabled: bool,
    colour: str | None = "green",
) -> Iterator[tqdm]:
    """Yield a bar to drive, hiding whether it's owned here or shared by the caller.

    If ``bar`` is given, it's reused as-is (a shared bar spanning multiple
    calls) and left open for the caller to close. Otherwise a bar is created
    here for the duration of the ``with`` block -- real if ``enabled``, a
    disabled (fully no-op) tqdm stand-in if not -- and always closed on exit.
    Either way the yielded bar is safe to call unconditionally: callers never
    need to branch on ownership or on whether progress display is on.
    """
    owns = bar is None
    if owns:
        bar = make_bar(desc, total, colour=colour, disable=not enabled)
    try:
        yield bar
    finally:
        bar_clear_item(bar)
        if owns:
            bar.close()
