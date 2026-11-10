"""
Streamlit shell over the generic chart sinks — driven entirely by an adapter.

This is the only module in BenchGrid that imports streamlit (a leaf
dependency: ``benchgrid.plots`` is not imported from ``benchgrid/__init__``).
A workload starts a viewer with its own tabs::

    from benchgrid.plots.app import render_panels, run
    from experiments.analysis.my_workload.adapter import adapter
    from experiments.analysis.my_workload.tabs import TABS

    run(adapter, TABS)

The shell owns the widgets only: a sidebar built from the adapter's declared
dimensions, then one streamlit tab per project-declared
:class:`~.adapter.Tab`.  Two renderers are offered as building blocks —
:func:`render_panels` (a dynamic grid of metric panels: metric × chart kind ×
coloring × direction, with a CSV download of the *same* aggregated frame it
plots), :func:`render_curve` (one tall curve plot per key combination, described
by a :class:`~.curves.CurvePlotSpec`) and :func:`render_pairs` (a tall trade-off
scatter of two metrics, described by a :class:`~.pairs.PairPlotSpec`), both with
their own metric dropdown and a CSV of the plotted frame.  Each renderer opens
with :func:`select_step` (the tab's step selector).  A tab may also ignore them
all and build its own view from :class:`~.context.TabContext`.

Aggregation and coloring are headless derives shared with any CLI
(:mod:`~.preprocess`, :mod:`~.curves`), so plots and exported CSVs can never
diverge.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import matplotlib.pyplot as plt
import polars as pl
import streamlit as st

from ..data import Matrix, _agg_expr, _filter_frame
from . import charts, preprocess
from .adapter import PlotsAdapter, Tab, default_load
from .coloring import COLOR_FIELD, Coloring, color_frame, slice_keys
from .context import TabContext
from .curves import LABEL_KEY, CurvePlotSpec, curve_frame
from .pairs import PairPlotSpec, pair_frame
from .preprocess import PLOT_COLOR, PLOT_LABEL, PLOT_MARKER

__all__ = [
    "available_metrics",
    "default_load",
    "load_dataset",
    "render_curve",
    "render_pairs",
    "render_panels",
    "run",
    "select_step",
]

#: Chart height (px) of a curve or trade-off tab — more room than a panel.
CURVE_HEIGHT = 512

DEFAULT_DPI = 2048

#: The workload's filter hook: it renders the tab's own filter widgets and
#: returns the row filters to apply to the plotted frame (e.g.
#: ``{"refinement": 5, "nelems": "005"}``).  The renderer calls it right below
#: its step selector, so the workload's filters see the selected step.
FilterHook = Callable[[TabContext], dict[str, Any]]


def select_step(
    ctx: TabContext,
    *,
    label: str | None = None,
    key: str = "tab",
    default: Any = None,
    values_filter: Callable[[str], bool] | None = None,
) -> Any:
    """
    Render the tab body's step selector and set ``ctx.step``.

    The selector offers every distinct ``adapter.step_key`` value of the loaded
    data (sorted, no "(all)": a view is always about one concrete step) and
    defaults to *default* when present, else ``adapter.step_default``, else the
    lowest value.  Returns the selected step, or ``None`` when the adapter
    declares no ``step_key`` or the column is empty.  *key* namespaces the
    widget per tab; *label* defaults to ``adapter.step_label`` so every view of
    one workload names its step dimension the same way.  *values_filter* (a
    ``value → bool`` predicate) narrows the offered values, e.g. one tab that
    lists only the generation steps.
    """
    adapter = ctx.adapter
    step_key = adapter.step_key
    if not step_key:
        ctx.step = None
        return None
    source = adapter.distinct_source or next(iter(ctx.dataset.sources))
    values = sorted(
        v
        for v in ctx.dataset.distinct(source, step_key)
        if v is not None and (values_filter is None or values_filter(v))
    )
    if not values:
        ctx.step = None
        return None
    if default is None:
        default = adapter.step_default
    index = values.index(default) if default in values else 0
    ctx.step = st.selectbox(
        label or adapter.step_label, values, index=index, key=f"step_{key}"
    )
    return ctx.step


def _series_default(values: list[Any], wanted: list[Any] | None) -> list[Any]:
    """
    Preselection of one series multiselect.

    *wanted* is the workload's declaration (``None`` = no declaration): the
    selected values are the declared ones that exist in *values*, so a
    declaration never filters a tree that lacks them.  When none of the
    declared values is loaded, everything stays selected (the filter bypasses).
    """
    if wanted is None:
        return values
    return [v for v in values if v in wanted] or values


def select_series(ctx: TabContext, *, key: str) -> None:
    """
    In-body series multiselects over ``adapter.series_keys`` (one per column).

    The sidebar dims are single-choice; this is the tab-local, multi-value
    counterpart every engine view opens with: each declared column offers the
    values its current sidebar slice contains (default: all, or the
    ``adapter.series_defaults`` preselection), and the selection is written
    back into ``ctx.filters``, where the ``is_in`` support of
    :func:`~benchgrid.data._filter_frame` narrows every ``ctx.frame`` read of
    this tab.  Like :func:`select_step` it is a no-op when the adapter declares
    nothing, so existing viewers are untouched.

    ``key`` namespaces the widgets (one selector per tab body, like the step
    selector); the per-tab ``filters`` copy made by :func:`run` keeps one
    tab's selection from leaking into the next.
    """
    keys = list(ctx.adapter.series_keys)
    if not keys:
        return
    dest = ctx.adapter.distinct_source or next(iter(ctx.dataset.sources))
    base = _filter_frame(ctx.dataset.frame(dest), ctx.filters)
    with st.container(horizontal=True):
        for col in keys:
            if col not in base.columns:
                continue
            values = base[col].unique().drop_nulls().sort().to_list()
            ctx.filters[col] = st.multiselect(
                col,
                values,
                default=_series_default(values, ctx.adapter.series_defaults.get(col)),
                key=f"series_{key}_{col}",
            )


def load_dataset(adapter: PlotsAdapter) -> Any:
    """Load (and cache) the dataset once per session, via ``adapter.load``."""
    loader = (
        adapter.load
        if adapter.load is not None
        else lambda: default_load(adapter)
    )
    return st.cache_resource(loader)()


def _sidebar(
    adapter: PlotsAdapter, ds
) -> tuple[dict[str, Any], str, list[str]]:
    """Sidebar widgets; returns (fixed filters, agg, x_keys)."""
    dest = adapter.distinct_source or next(iter(ds.sources))
    st.sidebar.header("Data filters")
    fixed: dict[str, Any] = {}
    for dim in adapter.dims:
        values = [v for v in ds.distinct(dest, dim) if v is not None]
        shown = ["(all)"] + [str(v) for v in values]
        selected = st.sidebar.selectbox(dim, shown)
        if selected != "(all)":
            # `shown` = ["(all)"] + [str(v) for v in values] → real values at 1+.
            fixed[dim] = values[shown.index(selected) - 1]

    x_keys = list(adapter.x_keys)
    if len(x_keys) > 1:
        label = adapter.compare_label or f"Compare {'·'.join(x_keys)} on x"
        if not st.sidebar.checkbox(label, value=True):
            x_keys = x_keys[:1]

    st.sidebar.divider()
    st.sidebar.header("Plot options")
    agg = st.sidebar.selectbox(adapter.agg_label, adapter.agg_methods)

    st.sidebar.divider()
    roots = ", ".join(str(d) for d in adapter.data_dirs) or "(custom loader)"
    st.sidebar.caption(f"**Loaded from** — {roots}")
    parts = []
    for source in ds.sources:
        frame = ds.frame(source)
        parts.append(f"{source}: {frame.height} rows × {len(frame.columns)} cols")
    st.caption(f"**Frames** — {'; '.join(parts)}")
    return fixed, agg, x_keys


def _panel_agg(
    ctx: TabContext, *, source: str, metric: str, x_keys: list[str], y_key: str
) -> pl.DataFrame:
    """Final-sample frame aggregated over (x_keys, y_key) with a combined x label."""
    grp = [*x_keys, y_key]
    frame = ctx.frame(source, final=True)
    out = frame.group_by(grp, maintain_order=True).agg(
        _agg_expr(ctx.agg, metric).alias(metric)
    )
    return preprocess.x_label(out, x_keys)


def _panel_chart(
    adapter: PlotsAdapter,
    kind: str,
    frame: pl.DataFrame,
    metric: str,
    color_field: str | None,
):
    chart_kind = kind or adapter.chart_kinds[0]
    if chart_kind == "3D static":
        return charts.bars_3d(
            frame, "x_label", adapter.y_key, metric, color_field=color_field
        )
    if chart_kind == "3D interactive":
        return charts.bars_3d_interactive(
            frame, "x_label", adapter.y_key, metric, color_field=color_field
        )
    return charts.heatmap(
        frame, "x_label", adapter.y_key, metric, color_field=color_field
    )


def _render_chart(chart, chart_kind: str) -> None:
    """Show the chart widget matching the kind returned by the panel."""
    if chart_kind == "3D static":
        st.pyplot(chart)
    elif chart_kind == "3D interactive":
        st.plotly_chart(chart, width="stretch")
    else:
        st.altair_chart(chart, width="stretch")


def _init_panel_state(adapter: PlotsAdapter) -> list[dict]:
    """Seed the session-state panel list once; returns the live list."""
    if "panels" not in st.session_state:
        defaults = [
            {
                "id": i + 1,
                "metric": panel.get("metric", adapter.metrics[0]),
                "chart_kind": panel.get("chart_kind", adapter.chart_kinds[0]),
                "coloring": panel.get("coloring", adapter.colorings[0]),
            }
            for i, panel in enumerate(adapter.default_panels)
        ]
        st.session_state["panels"] = defaults
    if "panel_seq" not in st.session_state:
        st.session_state["panel_seq"] = max(
            (p.get("id", 0) for p in st.session_state["panels"]), default=0
        )
    return st.session_state["panels"]


def _render_panel(ctx: TabContext, *, panel: dict, metrics: list[str]) -> bool:
    """Render one panel card; returns True when its Remove button was clicked."""
    adapter = ctx.adapter
    pid = panel["id"]
    metric = st.selectbox(
        "Metric",
        metrics,
        index=metrics.index(panel["metric"])
        if panel["metric"] in metrics
        else 0,
        key=f"panel_{pid}_metric",
    )
    chart_kind = st.radio(
        "Chart kind",
        adapter.chart_kinds,
        index=adapter.chart_kinds.index(
            panel.get("chart_kind", adapter.chart_kinds[0])
        ),
        horizontal=True,
        key=f"panel_{pid}_kind",
    )
    coloring_labels = [c.value for c in adapter.colorings]
    coloring_label = st.selectbox(
        "Coloring",
        coloring_labels,
        index=coloring_labels.index(
            Coloring(panel.get("coloring", adapter.colorings[0])).value
        ),
        key=f"panel_{pid}_coloring",
    )
    coloring = Coloring(coloring_label)
    direction = None
    if coloring.needs_direction:
        direction = st.selectbox(
            "Best direction",
            ["min", "max"],
            index=0 if adapter.best_of(metric) == "min" else 1,
            key=f"panel_{pid}_direction",
        )
    panel["metric"], panel["chart_kind"], panel["coloring"] = (
        metric,
        chart_kind,
        coloring,
    )

    source = adapter.metric_source[metric]
    st.caption(adapter.source_labels.get(source, source))
    base = _panel_agg(
        ctx,
        source=source,
        metric=metric,
        x_keys=ctx.x_keys,
        y_key=adapter.y_key,
    )
    if base.is_empty():
        st.info("No data for this panel with the current filters.")
        return st.button("Remove", key=f"panel_{pid}_remove")

    total = base.height
    if total > 1:
        kcol, dcol = st.columns(2)
        with kcol:
            keep = st.slider(
                "Points to show", 1, total, total, key=f"panel_{pid}_keep"
            )
        with dcol:
            side = st.radio(
                "Keep",
                ["best", "worst"],
                horizontal=True,
                key=f"panel_{pid}_keep_side",
            )
        if keep < total:
            # "best" is the adapter's best_of(metric) end; coloring and the CSV
            # below are computed on the kept rows, so the colors adapt.
            low_is_best = adapter.best_of(metric) == "min"
            base = base.sort(
                metric, descending=(side == "best") != low_is_best
            ).head(keep)

    colored = (
        color_frame(
            base,
            metric,
            coloring,
            group_keys=slice_keys(
                adapter.y_key, adapter.color_group_extras, base.columns
            ),
            direction=direction,
            version_key=adapter.version_key,
        )
        if coloring.needs_direction
        else None
    )

    c1, c2 = st.columns(2)
    with c1:
        if colored is None:
            pivot = base.select(["x_label", adapter.y_key, metric])
            matrix = Matrix.from_frame(
                pivot,
                rows="x_label",
                cols=adapter.y_key,
                metric=metric,
                title=f"{metric} ({ctx.agg})",
            )
            download_csv = matrix.to_csv()
            file_name = f"{metric}.csv"
        else:
            download_csv = colored.write_csv()
            file_name = f"{metric}_{coloring.suffix}.csv"
        st.download_button(
            "Matrix CSV",
            download_csv or "",
            file_name=file_name,
            mime="text/csv",
            key=f"panel_{pid}_csv",
        )
    with c2:
        remove = st.button("Remove", key=f"panel_{pid}_remove")

    chart = _panel_chart(
        adapter,
        chart_kind,
        colored if colored is not None else base,
        metric,
        COLOR_FIELD if colored is not None else None,
    )
    _render_chart(chart, chart_kind)
    return remove


def render_panels(
    ctx: TabContext,
    *,
    cols_per_row: int = 2,
    caption: str | None = None,
    step: bool = True,
    step_key: str = "panels",
) -> None:
    """
    Dynamic grid of metric panels from ``adapter.default_panels``.

    *cols_per_row* sets the grid width, *caption* replaces the default
    coloring explainer (``""`` hides it), and *step* (with *step_key*) toggles
    the body-header step selector.  Workloads whose filters live in the tab
    body call ``select_step`` themselves and pass ``step=False``.
    """
    adapter = ctx.adapter
    if step:
        select_step(ctx, key=step_key)
    select_series(ctx, key=step_key)
    panels = _init_panel_state(adapter)
    metrics = available_metrics(ctx)
    if not metrics:
        st.warning("No plottable metrics declared in the adapter.")
        return

    header = st.columns([1, 4])
    with header[0]:
        if st.button("＋ Add panel", key="add_panel"):
            st.session_state["panel_seq"] += 1
            panels.append(
                {
                    "id": st.session_state["panel_seq"],
                    "metric": metrics[0],
                    "chart_kind": adapter.chart_kinds[0],
                    "coloring": adapter.colorings[0],
                }
            )
    with header[1]:
        if caption is None:
            described = {
                Coloring.VALUE: "the metric-value colormap",
                Coloring.WINNER: "only the winning row of each slice keeps a color",
                Coloring.MINMAX: "every cell/bar is filled by its position within "
                "the slice (best green, worst red)",
            }
            caption = (
                "Each panel plots any metric from any source; its metric and one "
                f"of {len(adapter.chart_kinds)} chart kinds are independent. "
                "Colorings available here: "
                + "; ".join(
                    f"“{c.value}” — {described[c]}"
                    for c in adapter.colorings
                    if c in described
                )
                + "."
            )
        if caption:
            st.caption(caption)

    remove_ids: set[int] = set()
    for i in range(0, len(panels), cols_per_row):
        row = st.columns(cols_per_row)
        for j, col in enumerate(row):
            idx = i + j
            if idx >= len(panels):
                break
            with col:
                if _render_panel(ctx, panel=panels[idx], metrics=metrics):
                    remove_ids.add(panels[idx]["id"])
    if remove_ids:
        panels[:] = [p for p in panels if p["id"] not in remove_ids]


def available_metrics(ctx: TabContext) -> list[str]:
    """
    The adapter's declared metrics whose column exists in its source frame.

    A metric may be declared but absent from a scanned tree (e.g. an older
    sample), so both metric dropdowns offer what is actually plottable.
    """
    adapter = ctx.adapter
    keys = {source: ctx.dataset.keys(source) for source in ctx.dataset.sources}
    return [
        metric
        for metric in adapter.metrics
        if metric
        in keys.get(adapter.metric_source.get(metric, ""), frozenset())
    ]


def _spec_key(spec: CurvePlotSpec) -> str:
    """Stable widget-key stem for *spec* (identity + source + default metric)."""
    return "_".join(
        [spec.source, spec.metric, *spec.curve_keys, spec.select_best or ""]
    )


def render_curve(
    ctx: TabContext, spec: CurvePlotSpec, *, key: str | None = None
) -> None:
    """
    Plot *spec*'s curves and offer the plotted frame as CSV.

    Above the chart a "Metric" dropdown offers every declared metric of any
    source, like a panel: the derived frame follows the metric's source
    (``adapter.metric_source``) and, when the spec selects an envelope, its
    direction follows ``adapter.best_of(metric)``.  *spec.metric* /
    *spec.source* are the defaults.  The x axis is
    ``ctx.adapter.progress_key``; the aggregation over repeated measures is
    ``ctx.agg``, the step comes from the tab's step selector and the sidebar
    filters are already applied.  *key* only matters when the same spec is
    rendered in more than one tab.
    """
    adapter = ctx.adapter
    stem = f"curve_{key or _spec_key(spec)}"
    select_step(ctx, key=stem)
    select_series(ctx, key=stem)
    options = available_metrics(ctx)
    if spec.metric not in options:
        options.insert(0, spec.metric)
    metric = st.selectbox(
        "Metric",
        options,
        index=options.index(spec.metric),
        key=f"{stem}_metric",
    )
    if metric == spec.metric:
        plotted_spec = spec
    else:
        plotted_spec = replace(
            spec,
            metric=metric,
            source=adapter.metric_source.get(metric, spec.source),
            best=adapter.best_of(metric) if spec.select_best else spec.best,
        )

    frame = curve_frame(
        ctx.frame(plotted_spec.source),
        plotted_spec,
        progress_key=adapter.progress_key,
        agg=ctx.agg,
    )
    if frame.is_empty():
        st.info("No curve data with the current filters.")
        return
    plotted = frame.to_pandas()
    st.altair_chart(
        charts.lines_chart(
            plotted,
            x_key=adapter.progress_key,
            y_key=plotted_spec.metric,
            color_key=LABEL_KEY,
            text_key=plotted_spec.text_key,
            log_x=plotted_spec.log_x,
            log_y=plotted_spec.log_y,
            title="",
            height=CURVE_HEIGHT,
        ),
        width="stretch",
    )
    st.download_button(
        "Download curve data (CSV)",
        plotted.to_csv(index=False).encode(),
        file_name=f"{plotted_spec.metric}_curves.csv",
        mime="text/csv",
        key=f"{stem}_csv",
    )


def _pair_key(spec: PairPlotSpec) -> str:
    """Stable widget-key stem for *spec* (both axes + grouping + defaults)."""
    return "_".join(
        [
            spec.x_metric,
            spec.y_metric,
            *spec.point_keys,
            *spec.color_keys,
            spec.text_key or "",
        ]
    )


def _series_style(frame: pl.DataFrame, series: pl.DataFrame) -> dict[str, Any]:
    """
    Matplotlib kwargs for one series from the frame's ``plot_*`` columns.

    Empty when the frame carries no presentation columns; otherwise the
    series' ``plot_color`` / ``plot_marker`` (constant within the series, so
    the first non-null value wins).
    """
    style: dict[str, Any] = {}
    for column, kw in ((PLOT_COLOR, "color"), (PLOT_MARKER, "marker")):
        if column in frame.columns:
            values = series[column].drop_nulls()
            if len(values):
                style[kw] = values[0]
    return style


def _hull(
    xs: "Sequence[float]", ys: "Sequence[float]"
) -> tuple[list[float], list[float]] | None:
    """Monotone-chain convex hull; ``None`` when the cloud is a line or a point."""
    pts = sorted(set(zip(xs, ys)))
    if len(pts) < 3:
        return None

    def chain(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for p in points:
            while len(out) > 1 and (
                (out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])
            ) <= 0:
                out.pop()
            out.append(p)
        return out

    hull = chain(pts)[:-1] + chain(pts[::-1])[:-1]
    if len(hull) < 3:
        return None
    return [p[0] for p in hull], [p[1] for p in hull]


def _seed_region(
    group: pl.DataFrame, x_metric: str, y_metric: str, q: float = 0.1
) -> tuple[list[float], list[float]] | None:
    """
    Percentile area of one configuration's repeats: the hull of the seeds left
    inside the configuration's own ``q``..``1-q`` box on both axes, so an
    outlier seed on either axis cannot stretch the fill.  ``None`` when too
    few seeds survive to bound an area — the caller then draws a plain point,
    which is also what a single-seed dataset gives.

    *q* trims each axis separately, so the kept fraction is ``(1-2q)²`` for
    unrelated metrics — not ``1-2q``; the tab states that coverage on the plot.
    """
    if group.height < 3:
        return None
    xs, ys = group[x_metric], group[y_metric]
    # linear (not polars' "nearest" default): with a handful of seeds the
    # nearest-rank quantile returns a real sample, so the extremes survive.
    kept = group.filter(
        pl.col(x_metric).is_between(
            xs.quantile(q, interpolation="linear"),
            xs.quantile(1 - q, interpolation="linear"),
        )
        & pl.col(y_metric).is_between(
            ys.quantile(q, interpolation="linear"),
            ys.quantile(1 - q, interpolation="linear"),
        )
    )
    return _hull(kept[x_metric].to_list(), kept[y_metric].to_list())


def _draw_baseline(
    ax,
    baseline_frame: pl.DataFrame,
    *,
    x_metric: str,
    y_metric: str,
    point_keys: "Sequence[str]",
    quantile: float,
    grouped: bool,
    color: str,
    alpha: float,
    linestyle: str,
) -> None:
    """
    Draw the baseline series as reference: one light gray percentile area per
    configuration plus one dashed line at that configuration's mean *x*.

    Never one line per repeated measure — with seeds kept that buried the plot
    in dashes; the area above carries the spread instead.
    """
    groups = (
        baseline_frame.partition_by(point_keys, maintain_order=True)
        if point_keys
        else [baseline_frame]
    )
    for group in groups:
        region = (
            _seed_region(group, x_metric, y_metric, quantile) if grouped else None
        )
        if region is not None:
            ax.fill(
                *region,
                color=color,
                alpha=0.15,
                edgecolor="none",
                zorder=0,
            )
        ax.axvline(
            group[x_metric].mean(),
            color=color,
            alpha=alpha,
            linestyle=linestyle,
            zorder=0,
        )


def append_legend_keys(
    frame: pl.DataFrame,
    label_col: str,
    keys: "Sequence[str]",
) -> pl.DataFrame:
    """
    Rewrite the legend names as ``<label> (<key values>)``.

    Applied row-wise (missing columns are ignored), so a label shared by two
    workloads — e.g. two W-tree generations both running ``slide-bs`` — becomes
    two distinct series instead of one merged legend entry.
    """
    present = [k for k in keys if k in frame.columns]
    if not present or label_col not in frame.columns:
        return frame
    # Row-wise: the whole point is to *split* a label that two workloads share
    # (wtree and owtree both name a series "vSlide-BS"), so a per-label
    # mapping would collapse them again.
    suffix = pl.concat_str(
        [pl.col(k).cast(pl.Utf8) for k in present], separator=" "
    )
    return frame.with_columns(
        (pl.col(label_col) + " (" + suffix + ")").alias(label_col)
    )


def render_pairs_altair(
    ctx: TabContext,
    spec: PairPlotSpec,
    *,
    key: str | None = None,
    filters: FilterHook | None = None,
) -> None:
    """
    Plot *spec*'s paired metrics and offer the plotted frame as CSV.

    A step selector (tab body header) and one dropdown per axis over every
    declared metric of any source — an axis may cross sources — default to
    ``spec.y_metric`` / ``spec.x_metric``.  Each point is the final measure of
    the selected step, reduced over seeds by ``ctx.agg`` (see
    :func:`~.pairs.pair_frame`).

    *filters* is the workload's filter hook: it renders the tab's own filter
    widgets (right below the step selector) and returns the row filters to
    apply to the plotted frame.
    """
    stem = f"pairs_{key or _pair_key(spec)}"
    select_step(ctx, key=stem)
    extra = filters(ctx) if filters is not None else {}
    options = available_metrics(ctx)
    for metric in (spec.x_metric, spec.y_metric):
        if metric not in options:
            options.insert(0, metric)

    y_metric = st.selectbox(
        "Y metric",
        options,
        index=options.index(spec.y_metric),
        key=f"{stem}_y",
    )
    x_metric = st.selectbox(
        "X metric",
        options,
        index=options.index(spec.x_metric),
        key=f"{stem}_x",
    )
    if y_metric != spec.y_metric or x_metric != spec.x_metric:
        spec = replace(spec, y_metric=y_metric, x_metric=x_metric)

    frame = pair_frame(ctx, spec, agg=ctx.agg, filters=extra)
    if frame.is_empty():
        st.info("No paired data with the current filters.")
        return
    plotted = frame.to_pandas()
    st.altair_chart(
        charts.scatter(
            plotted,
            x_key=spec.x_metric,
            y_key=spec.y_metric,
            color_key=LABEL_KEY,
            text_key=spec.text_key,
            log_x=spec.log_x,
            log_y=spec.log_y,
            height=CURVE_HEIGHT,
        ),
        width="stretch",
    )
    st.download_button(
        "Download pair data (CSV)",
        plotted.to_csv(index=False).encode(),
        file_name=f"{spec.y_metric}_vs_{spec.x_metric}.csv",
        mime="text/csv",
        key=f"{stem}_csv",
    )


def render_pairs(
    ctx: TabContext,
    spec: PairPlotSpec,
    *,
    key: str | None = None,
    dpi: int | None = DEFAULT_DPI,
    fig_size: tuple[int] | None = None,
    filters: FilterHook | None = None,
    baseline_selector: str | None = None,
) -> None:
    """
    Plot *spec*'s paired metrics and offer the plotted frame as CSV.

    One scatter point per configuration, one series per color key.  When the
    plotting frame carries the workload's ``plot_*`` presentation columns
    (attached at load time; see :func:`~benchgrid.plots.preprocess.plot_columns`)
    every series is drawn with its own legend name (``plot_label``), color
    (``plot_color``) and marker (``plot_marker``); otherwise the generic
    ``_label`` identity and matplotlib's default marker are used.

    *filters* is the workload's filter hook: it renders the tab's own filter
    widgets (right below the step selector) and returns the row filters to
    apply to the plotted frame.

    The tab-local series multiselects of :func:`select_series` narrow this
    view like every other (via ``adapter.series_keys``).  *baseline_selector*
    names a frame column and adds the per-tab baseline selectbox: the chosen
    value is drawn as a mean reference line over a light gray percentile area
    (the ``spec.baseline_*`` behavior) instead of points, defaulting to
    *spec.baseline_value* ("(none)" = no baseline).
    """
    import textalloc

    results = charts.MatplotChartResultData()

    # == Setup ==
    stem = f"pairs_{key or _pair_key(spec)}"
    select_step(ctx, key=stem)
    select_series(ctx, key=stem)
    extra = filters(ctx) if filters is not None else {}

    options = available_metrics(ctx)
    for metric in (spec.x_metric, spec.y_metric):
        if metric not in options:
            options.insert(0, metric)
    with st.container(horizontal=True):
        y_metric = st.selectbox(
            "Y metric",
            options,
            index=options.index(spec.y_metric),
            key=f"{stem}_y",
        )
        x_metric = st.selectbox(
            "X metric",
            options,
            index=options.index(spec.x_metric),
            key=f"{stem}_x",
        )
        if y_metric != spec.y_metric or x_metric != spec.x_metric:
            spec = replace(spec, y_metric=y_metric, x_metric=x_metric)

    with st.container(horizontal=True):
        x_label = st.text_input("x-axis metric label", value="")
        y_label = st.text_input("y-axis metric label", value="")

    _default_width, _default_height = fig_size if fig_size else (5, 5)
    cont = st.container(horizontal=True)
    with cont:
        op_width = st.number_input(
            "Plot width:",
            min_value=0,
            max_value=20,
            value=_default_width,
            width=80,
        )
        op_height = st.number_input(
            "Plot height:",
            min_value=0,
            max_value=20,
            value=_default_height,
            width=80,
        )

        use_logx = st.checkbox("Use log-scale on X-axis", value=spec.log_x)
        use_logy = st.checkbox("Use log-scale on Y-axis", value=spec.log_y)

        # avoid_crossing_label_lines=True,
        # avoid_label_lines_overlap=True,
        op_avoid_crossing = st.checkbox(
            "Avoid crossing label lines", value=True
        )
        op_avoid_overlap = st.checkbox("Avoid label lines overlap", value=True)

        # A declared repeat key turns each configuration's seeds into one filled
        # percentile area instead of one point per seed (off = the flat view).
        regions = spec.keep_key is not None and st.checkbox(
            "Percentile areas (seed spread)",
            value=True,
            key=f"{stem}_regions",
        )
        # Trim per axis: q = the tail dropped on each side, so p q..p 1-q.
        quantile = (
            st.slider(
                "Percentile trim q (per axis)",
                min_value=0.0,
                max_value=0.25,
                value=0.1,
                step=0.01,
                format="%.2f",
                key=f"{stem}_q",
            )
            if regions
            else 0.0
        )
    if spec.keep_key and not regions:
        spec = replace(spec, keep_key=None)

    append_legend = False
    if spec.legend_keys:
        append_legend = st.checkbox(
            "Append " + " + ".join(spec.legend_keys) + " to legend names",
            value=False,
            key=f"{stem}_legend_keys",
        )

    # == Render ==
    frame = pair_frame(ctx, spec, agg=ctx.agg, filters=extra)
    # Drop rows with no data in either selected metric.
    frame = frame.drop_nulls([x_metric, y_metric])
    if frame.is_empty():
        st.info("No paired data with the current filters.")
        return
    if LABEL_KEY not in frame:
        st.error(f"Column '{LABEL_KEY}' not in plotting frame.")
        return

    # == Baseline selector (mechanics, when the caller opts in) ==
    # The series multiselects already narrowed this tab through ctx.filters
    # (select_series at body top), so the baseline offers what is plottable.
    if baseline_selector and baseline_selector in frame.columns:
        baseline_options = ["(none)"] + (
            frame[baseline_selector].unique().drop_nulls().sort().to_list()
        )
        index = (
            baseline_options.index(spec.baseline_value)
            if spec.baseline_value in baseline_options
            else 0
        )
        choice = st.selectbox(
            "Baseline struct (vertical lines)",
            baseline_options,
            index=index,
            key=f"{stem}_baseline",
        )
        if choice == "(none)":
            spec = replace(spec, baseline_value=None)
        else:
            spec = replace(spec, baseline_key=baseline_selector, baseline_value=choice)

    fig, ax = plt.subplots(figsize=(op_width, op_height))

    label_col = PLOT_LABEL if PLOT_LABEL in frame.columns else LABEL_KEY
    if append_legend:
        frame = append_legend_keys(frame, label_col, spec.legend_keys)
    labels = frame[label_col].unique()
    order = ctx.adapter.label_order
    if order:
        rank = {name: i for i, name in enumerate(order)}

        def _rank(name: str) -> int:
            # A legend name may carry the appended key suffix (see
            # append_legend_keys); rank it by its declared base label.
            if name in rank:
                return rank[name]
            for base, index in rank.items():
                if name.startswith(base + " ("):
                    return index
            return len(order)

        # Unlisted series keep their frame order behind the declared ones.
        labels = sorted(labels, key=_rank)

    baseline_frame = None
    if (
        spec.baseline_key
        and spec.baseline_value is not None
        and spec.baseline_key in frame.columns
    ):
        baseline_frame = frame.filter(
            pl.col(spec.baseline_key) == spec.baseline_value
        )
    else:
        baseline_frame = None

    if baseline_frame is not None and not baseline_frame.is_empty():
        _draw_baseline(
            ax,
            baseline_frame,
            x_metric=x_metric,
            y_metric=y_metric,
            point_keys=spec.point_keys,
            quantile=quantile,
            grouped=spec.keep_key is not None,
            color=spec.baseline_color,
            alpha=spec.baseline_alpha,
            linestyle=spec.baseline_linestyle,
        )
        labels_to_plot = [
            l for l in labels if l not in baseline_frame[label_col].to_list()
        ]
    else:
        labels_to_plot = labels

    results.texts = []
    results.markers = []
    results.colors = []
    grouped = spec.keep_key is not None
    for label in labels_to_plot:
        subdf = frame.filter(pl.col(label_col) == label)
        style = _series_style(frame, subdf)
        _color = style.get("color", "r")
        _marker = style.get("marker", "")
        # One artist per configuration when the repeats are kept (each gets a
        # percentile area), else the whole series at once, as before.
        groups = (
            subdf.partition_by(spec.point_keys, maintain_order=True)
            if grouped
            else [subdf]
        )
        for index, group in enumerate(groups):
            # One legend entry per series, not one per configuration.
            ax_label = label if index == 0 else None
            region = grouped and _seed_region(
                group, x_metric, y_metric, quantile
            )
            if region:
                hull_x, hull_y = region
                ax.fill(
                    hull_x, hull_y, color=_color, alpha=0.25,
                    edgecolor=_color, linewidth=0.8,
                    label=ax_label, zorder=1,
                )
                # Anchor the point label and the marker on the configuration's
                # middle: the seeds themselves hold no extra information here.
                points_x = [group[x_metric].median()]
                points_y = [group[y_metric].median()]
                ax.scatter(points_x, points_y, s=18, zorder=2, **style)
            else:
                points_x = group[x_metric]
                points_y = group[y_metric]
                ax.scatter(
                    points_x,
                    points_y,
                    label=ax_label,
                    s=60,
                    alpha=0.7,
                    **style,
                )

            results.x_points.extend(points_x)
            results.y_points.extend(points_y)

            # Store texts, color and marker for latter adjustment:
            if spec.text_key is not None:
                # NULL keys (canonical structs without a parameter) label
                # nothing; textalloc wants a string, not None.
                texts = group[spec.text_key].fill_null("")
                results.texts.extend(texts[:1] if region else texts)

            results.colors.extend([_color] * len(points_x))
            results.markers.extend([_marker] * len(points_x))

    # Transform before reallocating points text.
    if use_logx:
        ax.set_xscale("log")
    if use_logy:
        ax.set_yscale("log")

    if results.texts:
        textalloc.allocate(
            ax,
            results.x_points,
            results.y_points,
            results.texts,
            x_scatter=results.x_points,
            y_scatter=results.y_points,
            textsize=8,
            avoid_crossing_label_lines=op_avoid_crossing,
            avoid_label_lines_overlap=op_avoid_overlap,
            min_distance=0.008,
            max_distance=0.1,
            # direction="southeast"
            linecolor=results.colors or "r",
            textcolor=results.colors or "k",
        )

    labels = (x_label or x_metric, y_label or y_metric)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    # if title:
    #     ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    # plt.tight_layout()

    st.pyplot(fig, width="content")

    if regions:
        # The implication of q, stated where the reader looks at the result:
        # each axis keeps 1-2q of the seeds, and because the two clips are
        # independent the joint area is smaller (its bounds depend on whether
        # the metrics move together or trade off).
        axis_cover = 1 - 2 * quantile
        st.caption(
            f"Percentile areas: each axis keeps its central "
            f"p{quantile:.0%}–p{axis_cover + quantile:.0%} "
            f"({axis_cover:.0%} of the seeds). Both axes are clipped "
            f"independently, so the hull holds ≈ {axis_cover**2:.0%} of a "
            f"configuration's seeds when the metrics are unrelated, up to "
            f"{axis_cover:.0%} when they rise and fall together, down to "
            f"{max(0.0, 2 * axis_cover - 1):.0%} when they trade off."
        )

    _step_name = ctx.step
    _extra_tags = "_".join([f"{dim}-{val}" for dim, val in extra.items()])
    _extra_tags += "_" if _extra_tags else ""
    fname = f"{_step_name}_{_extra_tags}{spec.y_metric}-vs-{spec.x_metric}"

    with st.container(horizontal=True):
        st.download_button(
            "Download pair data (CSV)",
            frame.write_csv().encode(),
            file_name=f"{fname}.csv",
            mime="text/csv",
            key=f"{stem}_csv",
        )

        img = io.BytesIO()
        fig.savefig(img, format="pdf", dpi=dpi, bbox_inches="tight")
        st.download_button(
            label="Download image",
            data=img,
            file_name=f"{fname}.pdf",
            mime="application/pdf",
        )


def run(adapter: PlotsAdapter, tabs: Sequence[Tab]) -> None:
    """
    Run the viewer: load the data, build the sidebar, render *tabs*.

    Each :class:`~.adapter.Tab` renders into its own streamlit tab with the
    same :class:`~.context.TabContext`, so every tab sees the sidebar's current
    selection.
    """
    st.set_page_config(layout="wide", page_title=adapter.title)
    st.title(adapter.title)

    ds = load_dataset(adapter)
    if all(ds.frame(source).is_empty() for source in ds.sources):
        st.warning(
            "No data found under the adapter's data dirs. Point the adapter at a "
            "populated result tree."
        )
        st.stop()

    if not tabs:
        st.warning("No tabs declared for this adapter.")
        return

    fixed, agg, x_keys = _sidebar(adapter, ds)
    ctx = TabContext(
        adapter=adapter, dataset=ds, filters=fixed, agg=agg, x_keys=x_keys
    )
    for tab, spec in zip(st.tabs([tab.label for tab in tabs]), tabs):
        with tab:
            # Each tab renders with a private copy of the sidebar filters:
            # the in-body series selectors (``select_series``) write list
            # filters into ``ctx.filters`` and must not leak to the next tab.
            spec.render(replace(ctx, filters=dict(ctx.filters)))
