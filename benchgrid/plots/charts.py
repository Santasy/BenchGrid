"""
Generic chart sinks: a prepared Polars frame → a figure.

The three classic kinds are available for any (x_key, y_key, metric)
combination, plus a line sink for curve views (version lines, growth curves):

  heatmap             — returns altair.Chart          (2-D, interactive tooltips)
  bars_3d             — returns matplotlib.Figure     (3-D static, PNG/SVG)
  bars_3d_interactive — returns plotly.Figure         (3-D, rotate/zoom/hover)
  lines_chart         — returns altair.Chart          (2-D lines, optional labels)
  scatter             — returns altair.Chart          (2-D points, optional labels)

Data arrives **pre-filtered and aggregated** (see
:mod:`benchgrid.plots.preprocess` and the module docstring) as a Polars frame,
lazy frame, or flat ``list[dict]``.  The Polars → pandas conversion happens
here, at the boundary, as the very last mile.

The ``color_field`` seam keeps the two slice colorings workload-agnostic: a
frame decorated with ``_color`` (hex) by :mod:`benchgrid.plots.preprocess`
passes its per-cell colors straight through; without it the builders fall back
to a metric-driven colormap.  For 3-D the interactive bars use one Mesh3d
trace per bar (correct depth sorting with ``opacity=1``); ``x``/``y`` slots
map 1:1 to the sorted data values, same order as the heatmap.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import polars as pl

__all__ = ["heatmap", "bars_3d", "bars_3d_interactive", "lines_chart", "scatter"]


def _records(data: Any) -> list[dict]:
    """Coerce chart input to ``list[dict]``: dicts pass through, frames materialize."""
    if data is None:
        return []
    if isinstance(data, pl.LazyFrame):
        data = data.collect()
    if isinstance(data, pl.DataFrame):
        return data.to_dicts()
    if isinstance(data, pd.DataFrame):
        return data.to_dict("records")
    return list(data)


def _no_data(kind: str, title: str) -> Any:
    """An empty-state figure matching *kind*'s library."""
    if kind == "3D static":
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        fig.suptitle(title)
        return fig
    if kind == "3D interactive":
        import plotly.graph_objects as go  # noqa: PLC0415

        return go.Figure().update_layout(title=f"{title} — no data")
    import altair as alt  # noqa: PLC0415

    return (
        alt.Chart(pd.DataFrame({"msg": ["No data"]}))
        .mark_text(fontSize=14)
        .encode(text="msg:N")
        .properties(title=title)
    )


# ---------------------------------------------------------------------------
# Heatmap (Altair)
# ---------------------------------------------------------------------------


def heatmap(
    data: Any,
    x_key: str,
    y_key: str,
    metric: str,
    *,
    color_field: str | None = None,
    title: str = "",
    height: int = 350,
    scheme: str = "redyellowgreen",
    reverse: bool = True,
):
    """
    Generic heatmap: x_key × y_key cells, colored by *metric* or by *color_field*.

    With *color_field* (a ``_color`` hex decoration from ``preprocess``) each
    cell keeps its explicit color and no colorbar/legend is drawn — this is how
    the slice colorings (winner / min-max) are rendered.  Otherwise the cell
    color is the metric value through the *scheme* colormap (*reverse* keeps
    high values at the red end, like the legacy charts).
    """
    import altair as alt  # noqa: PLC0415

    records = _records(data)
    if not records:
        return _no_data("heatmap", title or metric)

    df = pd.DataFrame(records)
    df[x_key] = df[x_key].astype(str)
    y_num = pd.to_numeric(df[y_key], errors="coerce")
    df[y_key] = y_num if y_num.notna().all() else df[y_key].astype(str)

    tooltip = [
        alt.Tooltip(f"{x_key}:O"),
        alt.Tooltip(f"{y_key}:O"),
        alt.Tooltip(f"{metric}:Q", format=".4g"),
    ]
    if color_field is not None:
        hexes = sorted({str(r.get(color_field)) for r in records if r.get(color_field)})
        # never pass scale=None: Altair reads it as "no scale" (see _x_axis).
        color_kw: dict[str, Any] = (
            {"scale": alt.Scale(domain=hexes, range=hexes)} if hexes else {}
        )
        color = alt.Color(f"{color_field}:N", legend=None, **color_kw)
    else:
        color = alt.Color(
            f"{metric}:Q",
            scale=alt.Scale(scheme=scheme, reverse=reverse),
            title=metric,
        )

    return (
        alt.Chart(df, title=title or metric)
        .mark_rect()
        .encode(
            x=alt.X(f"{x_key}:O", title=x_key, sort=None),
            y=alt.Y(f"{y_key}:O", title=y_key, sort="ascending"),
            color=color,
            tooltip=tooltip,
        )
        .properties(height=height)
    )


# ---------------------------------------------------------------------------
# 3-D bar chart (matplotlib — static)
# ---------------------------------------------------------------------------


def _bar_colors(
    bar_records: list[dict],
    raw_vals: list[float],
    *,
    colormap: str,
    color_field: str | None,
) -> tuple[list[Any], Any, Any]:
    import matplotlib  # noqa: PLC0415
    import matplotlib.colors as mcolors  # noqa: PLC0415

    if color_field is not None:
        return [r.get(color_field) for r in bar_records], None, None
    cmap = matplotlib.colormaps[colormap]
    vmin, vmax = (min(raw_vals), max(raw_vals)) if raw_vals else (0, 1)
    return (
        [cmap(mcolors.Normalize(vmin=vmin, vmax=vmax)(v)) for v in raw_vals],
        None,
        None,
    )


def bars_3d(
    data: Any,
    x_key: str,
    y_key: str,
    metric: str,
    *,
    title: str = "",
    figsize: tuple[float, float] = (11, 7),
    colormap: str = "RdYlGn_r",
    color_field: str | None = None,
):
    """
    Generic 3-D bar chart via matplotlib ``bar3d()``.

    Returns a matplotlib Figure — display with ``st.pyplot()`` or save with
    ``fig.savefig("out.png", dpi=150, bbox_inches="tight")``.  Unlike Plotly
    Mesh3d, ``bar3d()`` renders every bar completely regardless of camera angle.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from mpl_toolkits.mplot3d import Axes3D  # noqa: PLC0415, F401 — '3d' projection

    records = _records(data)
    if not records:
        return _no_data("3D static", title or metric)

    try:
        x_vals = sorted({r[x_key] for r in records}, key=lambda v: (str(type(v)), v))
        y_vals = sorted({r[y_key] for r in records}, key=lambda v: (str(type(v)), v))
    except TypeError:
        x_vals = sorted({str(r[x_key]) for r in records})
        y_vals = sorted({str(r[y_key]) for r in records})

    x_labels = [str(v) for v in x_vals]
    y_labels = [str(v) for v in y_vals]
    x_idx = {v: i for i, v in enumerate(x_vals)}
    y_idx = {v: i for i, v in enumerate(y_vals)}

    bar_x, bar_y, bar_z, bar_dz, raw_vals, bar_records = [], [], [], [], [], []
    for r in records:
        xi, yi = x_idx.get(r[x_key]), y_idx.get(r[y_key])
        val = r.get(metric, 0)
        if xi is not None and yi is not None and val:
            bar_x.append(xi)
            bar_y.append(yi)
            bar_z.append(0)
            bar_dz.append(val)
            raw_vals.append(val)
            bar_records.append(r)

    colors, norm, cmap = _bar_colors(
        bar_records, raw_vals, colormap=colormap, color_field=color_field
    )

    dx = dy = 0.6
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(
        [xi - dx / 2 for xi in bar_x],
        [yi - dy / 2 for yi in bar_y],
        bar_z,
        dx,
        dy,
        bar_dz,
        color=colors,
        shade=True,
        alpha=0.96,
    )

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel(x_key, labelpad=10)
    ax.set_ylabel(y_key, labelpad=10)
    ax.set_zlabel(metric, labelpad=10)
    ax.set_title(title or metric, pad=14)

    if norm is not None:
        from matplotlib import cm  # noqa: PLC0415

        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        fig.colorbar(mappable, ax=ax, shrink=0.5, pad=0.1, label=metric)

    fig.subplots_adjust(left=0.05, right=0.85, bottom=0.1, top=0.92)
    return fig


# ---------------------------------------------------------------------------
# 3-D interactive bar chart (Plotly) — one Mesh3d trace per bar
# ---------------------------------------------------------------------------

#: Box face index sets — 12 triangles covering 6 faces
_BOX_I = [0, 0, 4, 4, 0, 0, 3, 3, 0, 0, 1, 1]
_BOX_J = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 2, 6]
_BOX_K = [2, 3, 6, 7, 5, 4, 6, 7, 7, 4, 6, 5]


def bars_3d_interactive(
    data: Any,
    x_key: str,
    y_key: str,
    metric: str,
    *,
    title: str = "",
    colormap: str = "RdYlGn_r",
    dx: float = 0.6,
    dy: float = 0.6,
    color_field: str | None = None,
):
    """
    Interactive 3-D bar chart: one Plotly Mesh3d trace per bar.

    Returns a ``plotly.graph_objects.Figure`` — display with
    ``st.plotly_chart()`` or export as HTML with ``fig.write_html("out.html")``.
    """
    import matplotlib  # noqa: PLC0415
    import matplotlib.colors as mcolors  # noqa: PLC0415
    import plotly.graph_objects as go  # noqa: PLC0415

    records = _records(data)
    if not records:
        return _no_data("3D interactive", title or metric)

    try:
        x_vals = sorted({r[x_key] for r in records}, key=lambda v: (str(type(v)), v))
        y_vals = sorted({r[y_key] for r in records}, key=lambda v: (str(type(v)), v))
    except TypeError:
        x_vals = sorted({str(r[x_key]) for r in records})
        y_vals = sorted({str(r[y_key]) for r in records})

    x_labels = [str(v) for v in x_vals]
    y_labels = [str(v) for v in y_vals]
    x_idx = {v: i for i, v in enumerate(x_vals)}
    y_idx = {v: i for i, v in enumerate(y_vals)}

    bars = []
    for r in records:
        xi, yi = x_idx.get(r[x_key]), y_idx.get(r[y_key])
        val = r.get(metric, 0)
        if xi is not None and yi is not None and val:
            bars.append((xi, yi, val, str(r[x_key]), str(r[y_key])))

    if not bars:
        return _no_data("3D interactive", f"{title or metric} — all zero")

    raw_vals = [b[2] for b in bars]
    vmin, vmax = min(raw_vals), max(raw_vals)
    cmap = matplotlib.colormaps[colormap]

    color_lookup = None
    if color_field is not None:
        color_lookup = {
            (str(r.get(x_key)), str(r.get(y_key))): r[color_field] for r in records
        }

    traces: list[Any] = []
    for xi, yi, h, x_lbl, y_lbl in bars:
        x0, x1 = xi - dx / 2, xi + dx / 2
        y0, y1 = yi - dy / 2, yi + dy / 2
        if color_field is not None and color_lookup is not None:
            color = color_lookup[(x_lbl, y_lbl)]
        else:
            color = mcolors.to_hex(cmap(mcolors.Normalize(vmin=vmin, vmax=vmax)(h)))
        traces.append(
            go.Mesh3d(
                x=[x0, x1, x1, x0, x0, x1, x1, x0],
                y=[y0, y0, y1, y1, y0, y0, y1, y1],
                z=[0, 0, 0, 0, h, h, h, h],
                i=_BOX_I,
                j=_BOX_J,
                k=_BOX_K,
                color=color,
                opacity=1.0,
                flatshading=True,
                hovertext=f"{x_lbl}<br>{y_lbl}<br>{metric}: {h:.4g}",
                hoverinfo="text",
                showscale=False,
                showlegend=False,
            )
        )

    # Invisible dummy trace that carries the colorbar (skipped when bars take
    # their color from *color_field* — there is no metric gradient to show).
    if color_field is None:
        traces.append(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(
                    colorscale=colormap,
                    color=[vmin, vmax],
                    cmin=vmin,
                    cmax=vmax,
                    showscale=True,
                    colorbar=dict(title=metric, thickness=16, len=0.7),
                    size=1,
                    opacity=0,
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title or metric,
        height=700,
        scene=dict(
            xaxis=dict(
                tickvals=list(range(len(x_labels))), ticktext=x_labels, title=x_key
            ),
            yaxis=dict(
                tickvals=list(range(len(y_labels))), ticktext=y_labels, title=y_key
            ),
            zaxis=dict(title=metric),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Line chart (Altair)
# ---------------------------------------------------------------------------


def _text_alt(df: pd.DataFrame, key: str):
    """Altair text field for *key*: integer, 3-digit float or nominal."""
    import altair as alt  # noqa: PLC0415

    if pd.api.types.is_integer_dtype(df[key]):
        return alt.Text(f"{key}:Q", format="d")
    if pd.api.types.is_float_dtype(df[key]):
        return alt.Text(f"{key}:Q", format=".3g")
    return alt.Text(f"{key}:N")


def _x_axis(key: str, log: bool):
    """Altair quantitative x axis for *key*, optionally log-scaled.

    ``scale`` is passed only when log-scaling: Altair reads an explicit
    ``scale=None`` as *no scale*, which drops the axis and draws the marks in
    raw data coordinates.
    """
    import altair as alt  # noqa: PLC0415

    kwargs: dict[str, Any] = {"title": key}
    if log:
        kwargs["scale"] = alt.Scale(type="log")
    return alt.X(f"{key}:Q", **kwargs)


def _y_axis(key: str, log: bool):
    """Altair quantitative y axis for *key*, optionally log-scaled (see :func:`_x_axis`)."""
    import altair as alt  # noqa: PLC0415

    kwargs: dict[str, Any] = {"title": key}
    if log:
        kwargs["scale"] = alt.Scale(type="log")
    return alt.Y(f"{key}:Q", **kwargs)


def lines_chart(
    data: Any,
    x_key: str,
    y_key: str,
    color_key: str,
    *,
    title: str = "",
    text_key: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
    height: int = 350,
):
    """
    Generic line chart: x_key × y_key, one line per *color_key*.

    With *text_key* each point is labeled with its value (e.g. the target size
    achieving a growth-curve envelope).  *log_x*/*log_y* log the axes;
    *height* is the figure height in pixels (curves want more than a heatmap).
    """
    import altair as alt  # noqa: PLC0415

    records = _records(data)
    if not records:
        return _no_data("heatmap", title or y_key)

    df = pd.DataFrame(records)
    if df.empty:
        return _no_data("heatmap", title or y_key)

    # The encodings below reference columns by name only, and Altair ships the
    # whole frame, so a wrong key renders a silent blank chart: check them here.
    missing = [key for key in (x_key, y_key, color_key) if key not in df.columns]
    if missing:
        raise KeyError(
            f"lines_chart: column(s) {missing} not in data; "
            f"available: {list(df.columns)}"
        )

    x = _x_axis(x_key, log_x)
    y = _y_axis(y_key, log_y)

    base = alt.Chart(df, title=title or y_key).properties(height=height)
    line = base.mark_line(point=alt.OverlayMarkDef(filled=True, size=70)).encode(
        x=x,
        y=y,
        color=alt.Color(f"{color_key}:N", title=color_key),
    )
    if text_key and text_key in df.columns:
        labels = base.mark_text(dy=-10, fontSize=10).encode(
            x=x, y=y, text=_text_alt(df, text_key), color=alt.Color(f"{color_key}:N")
        )
        return (line + labels).properties(height=height)
    return line


def scatter(
    data: Any,
    x_key: str,
    y_key: str,
    *,
    color_key: str | None = None,
    text_key: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
    height: int = 350,
    title: str = "",
    size: int = 70,
):
    """
    Generic scatter: x_key × y_key points, optionally one series per *color_key*.

    The trade-off shape (e.g. time vs bytes per key): the Pareto region is the
    lower-left cloud.  With *text_key* each point carries a label (e.g. the
    target node size).  Axes are numeric; *log_x*/*log_y* log them.
    """
    import altair as alt  # noqa: PLC0415

    records = _records(data)
    if not records:
        return _no_data("heatmap", title or f"{y_key} vs {x_key}")

    df = pd.DataFrame(records)
    if df.empty:
        return _no_data("heatmap", title or f"{y_key} vs {x_key}")

    x = _x_axis(x_key, log_x)
    y = _y_axis(y_key, log_y)

    base = alt.Chart(df, title=title).properties(height=height)
    mark = base.mark_point(filled=True, size=size, opacity=0.85)
    text = _text_alt(df, text_key) if text_key and text_key in df.columns else None

    if color_key:
        color = alt.Color(f"{color_key}:N", title=color_key)
        points = mark.encode(x=x, y=y, color=color)
        labels = (
            base.mark_text(dy=-10, fontSize=10).encode(x=x, y=y, text=text, color=color)
            if text is not None
            else None
        )
    else:
        points = mark.encode(x=x, y=y)
        labels = (
            base.mark_text(dy=-10, fontSize=10).encode(x=x, y=y, text=text)
            if text is not None
            else None
        )
    if labels is None:
        return points
    return (points + labels).properties(height=height)
