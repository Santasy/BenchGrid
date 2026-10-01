"""
Slice colorings — the *available types* of a panel view, as one enum.

A coloring is a **data-difference view**, not a rendering detail: each one is a
frame modifier (:mod:`.preprocess`) that adds the per-row fill the chart sinks
read from :data:`COLOR_FIELD`.  Adding a coloring means adding a member here and
a modifier in the registry below — the app shell never changes.

* :attr:`Coloring.VALUE` — the frame untouched; the sink maps *metric* to color
  (the metric-value colormap).
* :attr:`Coloring.WINNER` — only the best row per slice keeps a color
  (:func:`~.preprocess.winner_fields`); the rest get :data:`LOSER_COLOR`.
* :attr:`Coloring.MINMAX` — every row colored by its position within the slice
  (:func:`~.preprocess.mm_fields`), best end of the cmap first.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

import polars as pl

from .preprocess import mm_fields, winner_fields

__all__ = ["COLOR_FIELD", "Coloring", "color_frame", "slice_keys"]

#: Per-row fill column the chart sinks consume (``color_field=``) and the
#: colored CSVs carry.
COLOR_FIELD = "_color"

#: A frame modifier: (frame, metric, group_keys, direction) → colored frame.
_ColorFn = Callable[..., pl.DataFrame]


class Coloring(StrEnum):
    """The available slice colorings; the value is the UI label."""

    VALUE = "Gradient over metric"
    MINMAX = "Min-max gradient per slice"
    WINNER = "Only best metric per slice"

    @property
    def needs_direction(self) -> bool:
        """True when the best direction ("min"/"max") is part of the view."""
        return self is not Coloring.VALUE

    @property
    def suffix(self) -> str:
        """Downloaded CSV file-name suffix for this coloring."""
        return _SUFFIXES[self]


_SUFFIXES: dict[Coloring, str] = {
    Coloring.VALUE: "value",
    Coloring.MINMAX: "minmax",
    Coloring.WINNER: "winner",
}

#: The registry: coloring → frame modifier.
_MODIFIERS: dict[Coloring, _ColorFn] = {
    Coloring.VALUE: lambda df, metric, *, group_keys, direction: df,
    Coloring.MINMAX: mm_fields,
    Coloring.WINNER: winner_fields,
}


def slice_keys(y_key: str, extras: list[str], columns: list[str]) -> list[str]:
    """
    The slice a coloring is computed within: *y_key* plus the *extras* that
    are present in *columns* (the adapter's non-competing group columns).
    """
    return [y_key] + [key for key in extras if key != y_key and key in columns]


def color_frame(
    base: pl.DataFrame,
    metric: str,
    coloring: Coloring | str,
    *,
    group_keys: list[str],
    direction: str | None = None,
) -> pl.DataFrame:
    """
    Apply *coloring* to the aggregated *base* frame (the slice *group_keys*).

    Returns *base* unchanged for :attr:`Coloring.VALUE`, otherwise adds
    :data:`COLOR_FIELD`; *direction* defaults to "min".
    """
    modifier = _MODIFIERS[Coloring(coloring)]
    return modifier(
        base,
        metric,
        group_keys=group_keys,
        direction="min" if direction is None else direction,
    )
