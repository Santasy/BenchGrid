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

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import polars as pl
import streamlit as st

from ..data import Matrix, _agg_expr
from . import charts, preprocess
from .adapter import PlotsAdapter, Tab, default_load
from .coloring import COLOR_FIELD, Coloring, color_frame, slice_keys
from .context import TabContext
from .curves import LABEL_KEY, CurvePlotSpec, curve_frame
from .pairs import PairPlotSpec, pair_frame

__all__ = [
    "default_load",
    "render_curve",
    "render_panels",
    "render_pairs",
    "run",
    "select_step",
]

#: Chart height (px) of a curve or trade-off tab — more room than a panel.
CURVE_HEIGHT = 640


def select_step(
    ctx: TabContext, *, label: str = "Step", key: str = "tab", default: Any = None
) -> Any:
    """
    Render the tab body's step selector and set ``ctx.step``.

    The selector offers every distinct ``adapter.step_key`` value of the loaded
    data (sorted, no "(all)": a view is always about one concrete step) and
    defaults to *default* when present, else the lowest value.  Returns the
    selected step, or ``None`` when the adapter declares no ``step_key`` or the
    column is empty.  *key* namespaces the widget per tab.
    """
    adapter = ctx.adapter
    step_key = adapter.step_key
    if not step_key:
        ctx.step = None
        return None
    source = adapter.distinct_source or next(iter(ctx.dataset.sources))
    values = sorted(v for v in ctx.dataset.distinct(source, step_key) if v is not None)
    if not values:
        ctx.step = None
        return None
    index = values.index(default) if default in values else 0
    ctx.step = st.selectbox(label, values, index=index, key=f"step_{key}")
    return ctx.step


def _load_ds(adapter: PlotsAdapter) -> Any:
    """Load (and cache) the dataset once per session, via ``adapter.load``."""
    loader = adapter.load if adapter.load is not None else lambda: default_load(adapter)
    return st.cache_resource(loader)()


def _sidebar(adapter: PlotsAdapter, ds) -> tuple[dict[str, Any], str, list[str]]:
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
    counts = "; ".join(f"{s}: {ds.frame(s).height}" for s in ds.sources)
    st.caption(f"Records — {counts}")
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
        index=metrics.index(panel["metric"]) if panel["metric"] in metrics else 0,
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
        ctx, source=source, metric=metric, x_keys=ctx.x_keys, y_key=adapter.y_key
    )
    if base.is_empty():
        st.info("No data for this panel with the current filters.")
        return st.button("Remove", key=f"panel_{pid}_remove")

    colored = (
        color_frame(
            base,
            metric,
            coloring,
            group_keys=slice_keys(
                adapter.y_key, adapter.color_group_extras, base.columns
            ),
            direction=direction,
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


def render_panels(ctx: TabContext) -> None:
    """Dynamic grid of metric panels, two per row, from ``adapter.default_panels``."""
    adapter = ctx.adapter
    select_step(ctx, key="panels")
    panels = _init_panel_state(adapter)
    metrics = _available_metrics(ctx)
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
        described = {
            Coloring.VALUE: "the metric-value colormap",
            Coloring.WINNER: "only the winning row of each slice keeps a color",
            Coloring.MINMAX: "every cell/bar is filled by its position within "
            "the slice (best green, worst red)",
        }
        st.caption(
            "Each panel plots any metric from any source; its metric and one of "
            f"{len(adapter.chart_kinds)} chart kinds are independent. Colorings "
            "available here: "
            + "; ".join(
                f"“{c.value}” — {described[c]}"
                for c in adapter.colorings
                if c in described
            )
            + "."
        )

    remove_ids: set[int] = set()
    for i in range(0, len(panels), 2):
        row = st.columns(2)
        for j, col in enumerate(row):
            idx = i + j
            if idx >= len(panels):
                break
            with col:
                if _render_panel(ctx, panel=panels[idx], metrics=metrics):
                    remove_ids.add(panels[idx]["id"])
    if remove_ids:
        panels[:] = [p for p in panels if p["id"] not in remove_ids]


def _available_metrics(ctx: TabContext) -> list[str]:
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
        if metric in keys.get(adapter.metric_source.get(metric, ""), frozenset())
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
    options = _available_metrics(ctx)
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


def render_pairs(
    ctx: TabContext, spec: PairPlotSpec, *, key: str | None = None
) -> None:
    """
    Plot *spec*'s paired metrics and offer the plotted frame as CSV.

    A step selector (tab body header) and one dropdown per axis over every
    declared metric of any source — an axis may cross sources — default to
    ``spec.y_metric`` / ``spec.x_metric``.  Each point is the final measure of
    the selected step, reduced over seeds by ``ctx.agg`` (see
    :func:`~.pairs.pair_frame`).
    """
    stem = f"pairs_{key or _pair_key(spec)}"
    select_step(ctx, key=stem)
    options = _available_metrics(ctx)
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

    frame = pair_frame(ctx, spec, agg=ctx.agg)
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


def run(adapter: PlotsAdapter, tabs: Sequence[Tab]) -> None:
    """
    Run the viewer: load the data, build the sidebar, render *tabs*.

    Each :class:`~.adapter.Tab` renders into its own streamlit tab with the
    same :class:`~.context.TabContext`, so every tab sees the sidebar's current
    selection.
    """
    st.set_page_config(layout="wide", page_title=adapter.title)
    st.title(adapter.title)

    ds = _load_ds(adapter)
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
    ctx = TabContext(adapter=adapter, dataset=ds, filters=fixed, agg=agg, x_keys=x_keys)
    for tab, spec in zip(st.tabs([tab.label for tab in tabs]), tabs):
        with tab:
            spec.render(ctx)
