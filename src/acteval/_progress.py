"""tqdm progress-bar helpers for Evaluator's optional ``progress=True`` output."""

from collections.abc import Iterator
from contextlib import contextmanager

from tqdm import tqdm

_ITEM_WIDTH = 25


class _NullBar:
    """No-op stand-in for a disabled bar, so ``bar_scope`` never has to pay for
    a real ``tqdm`` instance when progress display is off."""

    _acteval_label = ""

    def update(self, n: int = 1) -> None:
        pass

    def set_description_str(self, desc: str | None = None, refresh: bool = True) -> None:
        pass

    def close(self) -> None:
        pass


_NULL_BAR = _NullBar()

Bar = tqdm | _NullBar


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


def bar_set_item(bar: Bar, item: str) -> None:
    field = f"{item:<{_ITEM_WIDTH}}"[:_ITEM_WIDTH]
    bar.set_description_str(f"{bar._acteval_label}  {field}", refresh=True)


def bar_clear_item(bar: Bar) -> None:
    # refresh=True: this is called right after a bar's loop finishes, so it's
    # also the only guaranteed redraw showing the bar at 100% before it may
    # sit untouched (and thus visually frozen) until a much later close().
    bar.set_description_str(f"{bar._acteval_label}  {'':>{_ITEM_WIDTH}}", refresh=True)


@contextmanager
def bar_scope(
    bar: Bar | None,
    desc: str,
    total: int,
    enabled: bool,
    colour: str | None = "green",
) -> Iterator[Bar]:
    """Yield a bar to drive, hiding whether it's owned here or shared by the caller.

    If ``bar`` is given, it's reused as-is (a shared bar spanning multiple
    calls) and left open for the caller to close. Otherwise a bar is created
    here for the duration of the ``with`` block -- a real ``tqdm`` if
    ``enabled``, or a shared ``_NullBar`` (no ``tqdm`` construction at all) if
    not -- and always closed on exit. Either way the yielded bar is safe to
    call unconditionally: callers never need to branch on ownership or on
    whether progress display is on.
    """
    owns = bar is None
    if owns:
        bar = make_bar(desc, total, colour=colour) if enabled else _NULL_BAR
    try:
        yield bar
    finally:
        bar_clear_item(bar)
        if owns:
            bar.close()
