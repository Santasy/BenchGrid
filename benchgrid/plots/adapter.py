"""
Workload adapter — the *declarations* that drive the generic plots engine.

A workload never touches chart code.  It fills a :class:`PlotsAdapter` with its
facts (metrics, dimensions, axes, run identity, default panels) and reuses the
whole view layer.  Only the load-time normalization that *reduces to a minimum
the project code* — project symbols, derived columns — stays project-side,
injected as ``enrich`` per source.

The adapter is plain data; the schema (folder/file layout) comes from JSON at
load time (``read_schemas``) and the frames are scanned Polars-native.  The
default :func:`default_load` materializes the :class:`~benchgrid.data.Dataset`;
a workload with special load logic (e.g. on-demand sources) replaces
``adapter.load`` instead.

The tabs are **not** declared here: a workload supplies its own
:class:`Tab` list to :func:`benchgrid.plots.app.run`, because what a tab shows
is project knowledge (see :mod:`~benchgrid.plots.context`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import polars as pl

from ..data import Dataset
from ..scan import scan_csv_lazy, scan_json
from ..schema import read_schemas
from .coloring import Coloring

if TYPE_CHECKING:
    from .context import TabContext

__all__ = ["PlotsAdapter", "Tab", "default_load"]

#: A source frame (possibly lazy) → enriched frame (symbols, derived columns).
EnrichFn = Callable[[pl.DataFrame | pl.LazyFrame], pl.DataFrame | pl.LazyFrame]


@dataclass
class Tab:
    """
    One project-declared tab: its label and the function that fills it.

    *render* receives a :class:`~benchgrid.plots.context.TabContext` (adapter,
    dataset, sidebar filters/agg/x keys) and is free to build any view from the
    generic blocks — :func:`~benchgrid.plots.app.render_panels`,
    :func:`~benchgrid.plots.app.render_curve` + a
    :class:`~benchgrid.plots.curves.CurvePlotSpec`, or its own Polars derives.
    """

    label: str
    render: Callable[["TabContext"], None]


@dataclass
class PlotsAdapter:
    """
    Everything the generic views need to know about one workload.

    Data/axes
    ---------
    *schema_file* + *data_dirs* locate the data (``read_schemas`` + the
    Polars scanners); *run_keys* is the run identity and *progress_key* the
    per-run progress column (``None`` = no final-sample step) whose maximum
    defines the comparable per-run value.  *step_key* names the batch-phase
    column each tab can switch with its body-header step selector (``None``
    = no selector).

    UI
    --
    *metrics* is the unified metric dropdown (any source), *metric_source*
    maps metric → source name, *source_metrics* the canonical per-source
    lists (Dataset introspection), *dims* the sidebar filters, *x_keys* the
    default x axis (multi → combined label), *y_key* the rows/color slice.

    Coloring
    --------
    *colorings* narrows the available panel views (default: every
    :class:`~.coloring.Coloring`); *color_group_extras* extends the per-slice
    coloring group (normally ``[y_key]``) with columns that must *not* compete
    in the slice, e.g. the outer struct key so winners are chosen per
    (struct, y).  *default_panels* pre-fills the metrics/compare panels tab.

    Hook
    ----
    ``enrich`` maps source name → frame normalization (project symbols/derived
    columns); ``load`` overrides :func:`default_load` wholesale if needed.

    Everything above is *data*; the tabs live in the workload (see :class:`Tab`).
    """

    title: str
    schema_file: str | Path
    data_dirs: list[str]
    metrics: list[str]
    metric_source: dict[str, str]
    source_metrics: dict[str, list[str]]
    run_keys: list[str]
    progress_key: str = "n"
    step_key: str | None = "step"
    dims: list[str] = field(default_factory=list)
    x_keys: list[str] = field(default_factory=lambda: ["struct_symbol", "version"])
    y_key: str = "node_bytes"
    color_group_extras: list[str] = field(default_factory=lambda: ["struct_symbol"])
    best_direction: dict[str, str] = field(default_factory=dict)
    agg_methods: list[str] = field(
        default_factory=lambda: ["mean", "min", "max", "median"]
    )
    chart_kinds: list[str] = field(
        default_factory=lambda: ["Heatmap", "3D interactive", "3D static"]
    )
    #: Available panel colorings (subset of :class:`~.coloring.Coloring`; the
    #: full set by default).  Labels/values come from the enum, not the adapter.
    colorings: list[Coloring] = field(default_factory=lambda: list(Coloring))
    default_panels: list[dict] = field(default_factory=list)
    source_labels: dict[str, str] = field(default_factory=dict)
    version_key: str = "version"
    agg_label: str = "Aggregation over seeds"
    compare_label: str = ""
    enrich: dict[str, EnrichFn] = field(default_factory=dict)
    load: Callable[[], "Any"] | None = None  # replaces default_load if set
    distinct_source: str | None = None  # source whose frame drives the sidebar

    def best_of(self, metric: str) -> str:
        """Best direction for *metric* ("min" unless the adapter says otherwise)."""
        return self.best_direction.get(metric, "min")


def _scan(source: str, schema: "Any", data_dirs: list[str]):
    if schema.file_pattern.format == "json":
        return scan_json(data_dirs, schema)
    return scan_csv_lazy(data_dirs, schema)


def default_load(adapter: PlotsAdapter):
    """
    Materialize the workload :class:`~benchgrid.data.Dataset` from *adapter*.

    Reads the JSON schema(s), scans every source with the matching Polars fast
    path, applies ``adapter.enrich[source]`` when provided, and wraps the
    frames in a :class:`~benchgrid.data.Dataset` with the declared metric keys.
    """
    schemas = read_schemas(adapter.schema_file)
    frames: dict[str, pl.DataFrame | pl.LazyFrame] = {}
    for name, schema in schemas.items():
        frame = _scan(name, schema, adapter.data_dirs)
        enrich = adapter.enrich.get(name)
        if enrich is not None:
            frame = enrich(frame)
        frames[name] = frame
    metric_keys = {name: frozenset(ms) for name, ms in adapter.source_metrics.items()}
    return Dataset(frames=frames, schemas=schemas, metric_keys=metric_keys)
