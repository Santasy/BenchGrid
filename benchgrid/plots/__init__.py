"""
Plotting layer for BenchGrid — generic views over a loaded :class:`Dataset`.

The large separation is **derive once, sink many**:

* *derive* (Polars → Polars) — filtering, final-sample extraction, seed
  reductions and the chart-related frame modifiers live in
  :mod:`benchgrid.plots.preprocess`, selected by the coloring types in
  :mod:`benchgrid.plots.coloring`.  Sinks get a plain frame back, never a
  figure.
* *sink* (frame → output) — :mod:`benchgrid.plots.charts` turns a prepared
  frame into one of the generic chart kinds (2-D heatmap, 3-D static bars,
  3-D interactive bars, line curves); :mod:`benchgrid.plots.app` is the
  streamlit shell that places those sinks on a panel grid driven by a
  :class:`benchgrid.plots.adapter.PlotsAdapter`.

Nothing here knows what a workload means.  A workload fills an adapter
(declarations: metrics, dims, axes, run identity), supplies its own tabs
(:class:`Tab` + :class:`TabContext`, rendered with the generic
:func:`~benchgrid.plots.app.render_panels` /
:func:`~benchgrid.plots.app.render_curve` blocks or its own derives) and reuses
the whole engine; its only remaining code is the load-time normalization
(project symbols, derived columns), injected via ``adapter.enrich``.

**Headless by construction:** :mod:`benchgrid.plots` is *not* imported from
``benchgrid/__init__``, so the query CLI and test suites never load altair,
matplotlib, plotly or streamlit.
"""

from . import charts, coloring, pairs, preprocess
from .adapter import PlotsAdapter, Tab, default_load
from .coloring import COLOR_FIELD, Coloring
from .context import TabContext
from .curves import CurvePlotSpec, curve_frame
from .pairs import PairPlotSpec, pair_frame
from .preprocess import LABEL_KEY

__all__ = [
    "COLOR_FIELD",
    "LABEL_KEY",
    "Coloring",
    "CurvePlotSpec",
    "PairPlotSpec",
    "PlotsAdapter",
    "Tab",
    "TabContext",
    "charts",
    "coloring",
    "curve_frame",
    "default_load",
    "pair_frame",
    "pairs",
    "preprocess",
]
