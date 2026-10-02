"""
Paired-metric plots — one point per configuration, x and y from *any* source.

The trade-off view: two metrics (e.g. wall time from one scan source and bytes
per key from another) plotted against each other, so the non-dominated
("Pareto") region is what a reader sees.  The derive is headless (Polars in →
Polars out) and reads through :class:`~.context.TabContext`, so it honours the
sidebar filters, the tab's step selector and the aggregation.

Both axes are reduced independently over the repeated measures (seeds) and then
inner-joined on the configuration keys: a configuration measured on only one
side has no pair and is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from ..data import _agg_expr, _filter_frame
from .context import TabContext
from .preprocess import LABEL_KEY, identity_label

__all__ = ["PairPlotSpec", "pair_frame"]


@dataclass
class PairPlotSpec:
    """
    What one paired plot shows — never what it is called (the tab names it).

    *x_metric* + *y_metric* are declared metrics (their sources resolve
    themselves, so an axis may cross sources); *point_keys* is what makes one
    point (normally the swept configurations, e.g. size × target node bytes);
    *color_keys* is what makes one series; *filters* are workload-declared row
    filters (applied only where the key exists); *text_key* the column labeling
    each point; *log_x*/*log_y* the axis scales.
    """

    x_metric: str
    y_metric: str
    point_keys: list[str]
    color_keys: list[str]
    #: Workload-declared row filters, applied per side where the key exists.
    filters: dict[str, Any] = field(default_factory=dict)
    #: Column labeling each point (e.g. the target node size).
    text_key: str | None = None
    log_x: bool = False
    log_y: bool = False


def pair_frame(
    ctx: TabContext, spec: PairPlotSpec, *, agg: str = "mean"
) -> pl.DataFrame:
    """
    One row per (*color_keys*, *point_keys*) with both axis metrics.

    Each side is taken as the final measure of the selected step
    (``ctx.frame(source, final=True)``), reduced over seeds by *agg* and joined
    on the configuration keys (inner: no pair, no point).  The result carries
    the keys, both metrics and :data:`~benchgrid.plots.preprocess.LABEL_KEY`.
    """
    adapter = ctx.adapter
    keys = list(dict.fromkeys([*spec.color_keys, *spec.point_keys]))
    unknown = [
        m for m in (spec.x_metric, spec.y_metric) if m not in adapter.metric_source
    ]
    if unknown:
        raise ValueError(
            f"PairPlotSpec metrics not declared in adapter.metric_source: {unknown}"
        )

    def side(metric: str) -> pl.DataFrame:
        frame = ctx.frame(adapter.metric_source[metric], final=True)
        if spec.filters:
            frame = _filter_frame(
                frame, {k: v for k, v in spec.filters.items() if k in frame.columns}
            )
        return frame.group_by(keys, maintain_order=True).agg(
            _agg_expr(agg, metric).alias(metric)
        )

    pairs = side(spec.x_metric).join(side(spec.y_metric), on=keys, how="inner")
    return pairs.with_columns(identity_label(spec.color_keys).alias(LABEL_KEY)).sort(
        keys
    )
