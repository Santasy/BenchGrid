"""
Batched curve plots — one curve per unique key combination, x = progress.

This is the *derive* half of a curve view (Polars in → Polars out, no
streamlit): take the rows of one source, keep the project's declared rows
(*spec.filters*), reduce the repeated measures of each batch point, and shape
the result as a long frame ready for :func:`~.charts.lines_chart`.

The generic contract is identity + selection only:

* **identity** — *curve_keys*; every unique combination is one curve, and the
  engine writes the combined :data:`LABEL_KEY` from them.
* **selection** — *select_best*; one configuration key (e.g. the target node
  size) whose best value per progress point survives, so the curve is the
  *envelope* over configurations.  Workloads with several competing keys keep
  them all in *curve_keys* instead (no selection).

Everything else is project knowledge and therefore lives in *spec* — declared
in the workload, never inferred here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from ..data import _agg_expr, _filter_frame
from .preprocess import LABEL_KEY, identity_label

__all__ = ["CurvePlotSpec", "curve_frame"]


@dataclass
class CurvePlotSpec:
    """
    What one curve plot shows — never what it is called (the tab names it).

    *source* + *metric* pick the default frame and y channel; *curve_keys* the
    curve identity; *select_best* + *best* the optional envelope over one
    configuration key; *filters* the workload's own row filters (e.g. one batch
    phase); *text_key* the column labeling each point (typically the
    configuration that won); *log_x*/*log_y* the axis scales.

    A view may widen both defaults: :func:`~.app.render_curve` lets the user
    pick any declared metric and then resolves the source itself.
    """

    source: str
    metric: str
    curve_keys: list[str]
    #: Configuration key whose best value per progress point is kept.
    select_best: str | None = None
    #: Direction of that selection ("min" keeps the smallest *metric*).
    best: str = "min"
    #: Workload-declared row filters, applied before any reduction.
    filters: dict[str, Any] = field(default_factory=dict)
    #: Column labeling each point (e.g. the winning configuration).
    text_key: str | None = None
    log_x: bool = False
    log_y: bool = False


def curve_frame(
    df: pl.DataFrame,
    spec: CurvePlotSpec,
    *,
    progress_key: str,
    agg: str = "mean",
) -> pl.DataFrame:
    """
    Shape *df* into the long frame of *spec*'s curve plot.

    Filters with *spec.filters*, reduces the repeated measures of every
    (curve, progress, configuration) point with *agg*, keeps the *best*
    configuration per point when *spec.select_best* is set (ties are **not**
    broken — every configuration equal to the best survives), and writes
    :data:`LABEL_KEY`.  Result columns: the curve keys, *progress_key*,
    *spec.metric*, the surviving configuration and :data:`LABEL_KEY`, sorted by
    identity and progress.
    """
    frame = _filter_frame(df, spec.filters)
    curve_keys = list(spec.curve_keys)
    group = [*curve_keys, progress_key]
    if spec.select_best is not None:
        group.append(spec.select_best)

    reduced = frame.group_by(group, maintain_order=True).agg(
        _agg_expr(agg, spec.metric).alias(spec.metric)
    )
    if spec.select_best is not None:
        best = (
            pl.col(spec.metric).min()
            if spec.best == "min"
            else pl.col(spec.metric).max()
        )
        reduced = reduced.filter(
            pl.col(spec.metric) == best.over([*curve_keys, progress_key])
        )

    label = identity_label(curve_keys)
    return reduced.with_columns(label.alias(LABEL_KEY)).sort(
        [*curve_keys, progress_key]
    )
