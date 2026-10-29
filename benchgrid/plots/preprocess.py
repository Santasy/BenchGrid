"""
Headless frame modifiers for the plots layer (Polars in → Polars out).

These are the *derive* half of the "derive once, sink many" split: they take
an already-filtered/aggregated frame and add the columns the chart sinks and
savable CSVs need.  No figure is produced here, so a CLI can export the exact
same frames an app panel plots.

Generic preparation lives here (:func:`x_label`, :func:`identity_label` — the
combined identity column the curve/pair views share); the *coloring* modifiers
are the slice views selected by :class:`~.coloring.Coloring`:
:func:`winner_fields` ("best version per slice") and :func:`mm_fields`
("min-max per slice"), both parameterized by *group_keys* — normally the
y/slice dimension(s) — and *direction*.

Workload-declared presentation is read through the ``plot_*`` convention
(:data:`PLOT_LABEL`, :data:`PLOT_COLOR`, :data:`PLOT_MARKER`): frames
decorated with those columns at load time override the generic series label,
color and marker in the matplotlib sinks — see :func:`plot_columns`.
"""

from __future__ import annotations

from typing import Any

import polars as pl

#: Column the curve/pair derives write: the identity keys joined with "·".
LABEL_KEY = "_label"

#: Workload-declared per-series presentation — the ``plot_*`` convention
#: (kept name-compatible with the legacy full-routine frames).  A frame
#: carrying them gets its own legend name, color and marker in the matplotlib
#: sinks; without them the sinks fall back to the generic series label.
PLOT_LABEL: str = "plot_label"
PLOT_COLOR: str = "plot_color"
PLOT_MARKER: str = "plot_marker"
#: The presentation columns, in the order they are checked/carried.
PLOT_COLUMNS: tuple[str, ...] = (PLOT_LABEL, PLOT_COLOR, PLOT_MARKER)

#: Version palette used by :func:`winner_fields` for the winning cells.
VERSION_COLORS: list[str] = [
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#f7cb59",
    "#d0576b",
]

#: Fill for non-winning cells/bars.
LOSER_COLOR = "#d3d3d3"

#: Slice min-max colormap (0 → green/best, 1 → red/worst).
SLICE_CMAP = "RdYlGn_r"


def x_label(df: pl.DataFrame, x_keys: list[str], *, sep: str = "·") -> pl.DataFrame:
    """Add a combined ``x_label`` column from *x_keys* (e.g. struct·version)."""
    return df.with_columns(
        pl.concat_str([pl.col(k).cast(pl.Utf8) for k in x_keys], separator=sep).alias(
            "x_label"
        )
    )


def identity_label(keys: list[str], *, sep: str = "·") -> pl.Expr:
    """
    Expression joining *keys* into one string column (:data:`LABEL_KEY`).

    The shared identity convention of the curve and pair derives: one line or
    series per unique combination of its keys, named by that combination.
    """
    return pl.concat_str([pl.col(key).cast(pl.Utf8) for key in keys], separator=sep)


def plot_columns(df: Any) -> list[str]:
    """
    The ``plot_*`` presentation columns present in *df*.

    Non-empty only when the workload decorated its frames with the series
    presentation (e.g. at load time); the derives carry them through and the
    matplotlib sinks honour them (see :data:`PLOT_COLUMNS`).
    """
    return [c for c in PLOT_COLUMNS if c in df.columns]


def winner_fields(
    df: pl.DataFrame,
    metric: str,
    *,
    group_keys: list[str],
    direction: str = "min",
    palette: list[str] | None = None,
    version_key: str,
) -> pl.DataFrame:
    """
    Add ``_winner`` (best metric per *group_keys* slice) and ``_color``
    (a distinct palette color per *version_key* value for winners,
    :data:`LOSER_COLOR` otherwise).

    *version_key* is required: it names the column whose distinct values share
    the palette among winners.  When the column is absent from *df* every row
    gets :data:`LOSER_COLOR` (no winner can be distinguished).
    """
    best = pl.col(metric).min() if direction == "min" else pl.col(metric).max()
    best_val = df.group_by(group_keys).agg(best.alias("_best"))
    out = df.join(best_val, on=group_keys).with_columns(
        (pl.col(metric) == pl.col("_best")).alias("_winner")
    )
    palette = palette or VERSION_COLORS
    if version_key in out.columns:
        versions = {
            v: i for i, v in enumerate(dict.fromkeys(out[version_key].to_list()))
        }
        win_color = pl.col(version_key).replace_strict(
            {v: palette[i % len(palette)] for v, i in versions.items()},
            default=LOSER_COLOR,
        )
    else:
        win_color = pl.lit(LOSER_COLOR)
    return out.with_columns(
        pl.when(pl.col("_winner"))
        .then(win_color)
        .otherwise(pl.lit(LOSER_COLOR))
        .alias("_color")
    )


def mm_fields(
    df: pl.DataFrame,
    metric: str,
    *,
    group_keys: list[str],
    direction: str = "min",
    cmap: str = SLICE_CMAP,
    version_key: str = "",
) -> pl.DataFrame:
    """
    Add ``_norm`` (0 = best within each *group_keys* slice, 1 = worst) and
    ``_color`` (slice cmap hex for that position).

    *version_key* is unused here (the slice itself carries the color); it keeps
    the modifier signature uniform with :func:`winner_fields`.
    """
    import matplotlib
    import matplotlib.colors as mcolors

    lo = pl.col(metric).min().over(group_keys)
    hi = pl.col(metric).max().over(group_keys)
    norm = (pl.col(metric) - lo) / (hi - lo)
    if direction == "max":
        norm = 1.0 - norm
    data = df.with_columns(norm.fill_null(0.0).clip(0.0, 1.0).alias("_norm"))
    colormap = matplotlib.colormaps[cmap]

    def _hex(t: float) -> Any:
        return mcolors.to_hex(colormap(float(t)))

    return data.with_columns(
        pl.col("_norm").map_elements(_hex, return_dtype=pl.Utf8).alias("_color")
    )


__all__ = [
    "LABEL_KEY",
    "PLOT_COLOR",
    "PLOT_COLUMNS",
    "PLOT_LABEL",
    "PLOT_MARKER",
    "identity_label",
    "mm_fields",
    "plot_columns",
    "winner_fields",
    "x_label",
]
