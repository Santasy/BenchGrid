"""
Dataset and Matrix — the BenchGrid query surface over scanned Polars frames.

``Dataset`` wraps named frames (one per :class:`benchgrid.schema.ScanSchema`
source) and answers "questions" about the data: filtering, dimension
introspection, final-sample extraction, reductions over dimensions and pivot matrices.
Everything in the query pipeline stays DataFrame-native — filters, group-bys,
pivots and reductions run in Polars.

- :meth:`Dataset.final_sample` — the last measure per run (e.g. the maximum
  ``n`` row of a cumulative sampler), the comparable per-run value.
- :meth:`Dataset.aggregate` — reduce one metric by any method over group keys.
- :meth:`Dataset.matrix` — a rows x cols pivot of one metric, renderable as
  aligned terminal text or CSV (:class:`Matrix`).

``Matrix`` is the terminal/CSV sink: ``to_text`` prints an aligned table and
``to_csv`` writes the pivot. This is the "common output (terminal/tables/csv)" path.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import polars as pl

from .schema import ScanSchema

__all__ = ["Dataset", "Matrix"]


def _to_frame(data: Any) -> pl.DataFrame:
    """Normalize ``records | frame | lazy frame | None`` to a DataFrame."""
    if data is None:
        return pl.DataFrame(schema=[])
    if isinstance(data, pl.DataFrame):
        return data
    if isinstance(data, pl.LazyFrame):
        return data.collect()
    if isinstance(data, (list, tuple, dict)):
        return pl.DataFrame(data)
    raise TypeError(
        f"expected list[dict] | polars.DataFrame | polars.LazyFrame, got {type(data).__name__}"
    )


def _filter_frame(df: pl.DataFrame, fixed: dict[str, Any]) -> pl.DataFrame:
    """
    Exact-match filter with string coercion on both sides (legacy semantics:
    ``fixed={"version": "simple"}`` matches an int column value ``0`` only by
    string equality — columns are compared after casting both sides to str).
    Unknown keys are ignored, like ``aggregate.filter_records``.
    """
    for key, value in fixed.items():
        if key in df.columns:
            df = df.filter(pl.col(key).cast(pl.Utf8) == str(value))
    return df


def _agg_expr(method: str, metric: str) -> pl.Expr:
    """Polars reduction expression per aggregate method name."""
    expr = {
        "min": pl.col(metric).min(),
        "max": pl.col(metric).max(),
        "mean": pl.col(metric).mean(),
        "median": pl.col(metric).median(),
    }.get(method)
    if expr is None:
        raise ValueError(f"unknown method {method!r} (use min/max/mean/median)")
    return expr


class Dataset:
    """
    Polars-backed records grouped by source name + the schemas that produced
    them.

    Parameters
    ----------
    frames       : mapping of source name → :class:`polars.DataFrame` (or
                   :class:`polars.LazyFrame`, collected eagerly).
    schemas      : optional mapping of source name → :class:`ScanSchema`.
    metric_keys  : canonical set of measurement columns per source, e.g.
                   ``{"time": {"time_us"}, "memory": {"height"}}`` — used only
                   for introspection (``dimensions()`` / ``metrics()``).
    """

    def __init__(
        self,
        frames: dict[str, pl.DataFrame | pl.LazyFrame],
        schemas: dict[str, ScanSchema] | None = None,
        metric_keys: dict[str, frozenset[str]] | None = None,
    ):
        self.frames: dict[str, pl.DataFrame] = {
            name: _to_frame(frame) for name, frame in frames.items()
        }
        self.schemas = schemas or {}
        self.metric_keys = metric_keys or {}

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------

    @property
    def sources(self) -> list[str]:
        """Source names in insertion order."""
        return list(self.frames)

    def frame(self, source: str) -> pl.DataFrame:
        """The eager Polars frame for *source*."""
        if source not in self.frames:
            raise KeyError(source)
        return self.frames[source]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def keys(self, source: str) -> set[str]:
        """All frame column names for *source*."""
        return set(self.frame(source).columns)

    def metrics(self, source: str) -> list[str]:
        return sorted(self.metric_keys.get(source, frozenset()))

    def dimensions(self, source: str) -> list[str]:
        """Frame columns that are not known metrics (dimensions)."""
        return sorted(
            set(self.frame(source).columns) - self.metric_keys.get(source, frozenset())
        )

    def distinct(self, source: str, key: str) -> list[Any]:
        """Distinct values for a dimension, in first-seen order."""
        return self.frame(source).get_column(key).unique(maintain_order=True).to_list()

    # ------------------------------------------------------------------
    # Queries (polars-native — return DataFrames)
    # ------------------------------------------------------------------

    def filter(self, source: str, **fixed: Any) -> pl.DataFrame:
        """Rows of *source* matching *fixed* (string-compared), as a frame."""
        return _filter_frame(self.frame(source), fixed)

    def final_sample(self, source: str, by: list[str], on: str) -> pl.DataFrame:
        """
        The last measure per run: rows where *on* equals its maximum within the
        run identity *by*.

        For cumulative samplers this is the comparable per-run value (e.g. the
        final ``n`` of step 1).  Equivalent to::

            df.filter(pl.col(on) == pl.col(on).max().over(*by))
        """
        df = self.frame(source)
        return df.filter(pl.col(on) == pl.col(on).max().over(*by))

    def aggregate(
        self,
        source: str,
        group_keys: list[str],
        metric: str,
        method: str = "mean",
    ) -> pl.DataFrame:
        """Reduce *metric* with *method* across *group_keys* → DataFrame."""
        return (
            self.frame(source)
            .group_by(group_keys, maintain_order=True)
            .agg(_agg_expr(method, metric).alias(metric))
        )

    def matrix(
        self,
        source: str,
        rows: str,
        cols: str,
        metric: str,
        method: str = "mean",
        *,
        label: str | None = None,
    ) -> "Matrix":
        """
        Pivot *metric* as rows(rows) × columns(cols).

        All other dimensions present in the frame are collapsed by *method*
        (e.g. ``mean`` reduces over seeds to the per-struct value).
        """
        reduced = self.aggregate(source, [rows, cols], metric, method)
        return Matrix.from_frame(
            reduced, rows, cols, metric, title=label or f"{metric} ({method})"
        )

    def to_csv(self, source: str, output: str | Path | TextIO | None = None):
        """
        Write a source frame as CSV (Polars-native; header row included).

        With *output* = a path or open file, writes there and returns None.
        With *output* = None, returns the CSV as bytes.
        """
        buf = io.StringIO()
        with buf:
            self.frame(source).write_csv(buf)
        csv_bytes = buf.getvalue().encode()

        if output is None:
            return csv_bytes
        if isinstance(output, (str, Path)):
            Path(output).write_bytes(csv_bytes)
            print(f"[benchgrid] wrote {output}")
        else:
            output.write(buf.getvalue())
        return None


@dataclass
class Matrix:
    """
    A rows(rows) x columns(cols) pivot of one metric.

    ``values`` maps (row_value, col_value) → number.  Row and column ordering
    follows a deterministic sort (numeric first, then string), which is an
    intentional improvement over scan-order pivots.  Renders to aligned
    terminal text (:meth:`to_text`) or CSV (:meth:`to_csv`); pandas is only
    produced at the chart boundary (:meth:`to_dataframe`).
    """

    rows: str
    cols: str
    metric: str
    values: dict[tuple[Any, Any], float]
    title: str = ""

    @classmethod
    def from_frame(
        cls,
        df: pl.DataFrame,
        rows: str,
        cols: str,
        metric: str,
        *,
        title: str = "",
    ) -> "Matrix":
        """Build a Matrix from an aggregated frame with columns [rows, cols, metric]."""
        df = df.select([rows, cols, metric]).sort([rows, cols])
        values = {(r[rows], r[cols]): r[metric] for r in df.to_dicts()}
        return cls(rows=rows, cols=cols, metric=metric, values=values, title=title)

    def row_values(self) -> list[Any]:
        """Row values in matrix order."""
        seen: dict[Any, None] = {}
        for r, _ in self.values:
            seen.setdefault(r, None)
        return list(seen)

    def col_values(self) -> list[Any]:
        """Column values in matrix order."""
        seen: dict[Any, None] = {}
        for _, c in self.values:
            seen.setdefault(c, None)
        return list(seen)

    def to_dataframe(self):
        """Pivot as a pandas DataFrame (row label = row value, columns = col values)."""
        import pandas as pd  # noqa: PLC0415 — optional, chart/export boundary only

        rows_data: dict[str, dict[str, float]] = {}
        for rv in self.row_values():
            rows_data[rv] = {
                cv: self.values.get((rv, cv), float("nan")) for cv in self.col_values()
            }
        return pd.DataFrame.from_dict(rows_data, orient="index").rename_axis(self.rows)

    @staticmethod
    def _fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def to_text(self) -> str:
        """
        Render an aligned text table::

            height (mean)                        # title
            struct_symbol      64        128       256
            ---------------------------------------
            w                 8.19       7.36      7.83
        """
        rows = self.row_values()
        cols = self.col_values()
        cells = [
            [self._fmt(self.values.get((rv, cv), float("nan"))) for cv in cols]
            for rv in rows
        ]
        value_width = max((len(c) for row in cells for c in row)) if cells else 8
        value_width = max(value_width, len(self.metric) + 2)

        row_name = self.rows
        header_pad = max(len(row_name), max((len(str(r)) for r in rows), default=0)) + 2

        lines = [self.title, ""]
        head = f"{row_name:<{header_pad}}" + "".join(
            f"{self._fmt(c):>{value_width}}" for c in cols
        )
        lines.append(head)
        lines.append("-" * len(head))
        for rv, cell in zip(rows, cells):
            lines.append(
                f"{rv:<{header_pad}}" + "".join(f"{c:>{value_width}}" for c in cell)
            )
        return "\n".join(lines)

    def to_csv(
        self,
        output: str | Path | TextIO | None = None,
    ) -> bytes | None:
        """
        Write the pivot as CSV (one header row: ``rows`` + column values).

        With *output* = a path or open file, writes there and returns None.
        With *output* = None, returns the CSV as bytes.
        """
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([self.rows] + self.col_values())
        for rv in self.row_values():
            writer.writerow(
                [rv] + [self.values.get((rv, cv), "") for cv in self.col_values()]
            )
        csv_bytes = buf.getvalue().encode()

        if output is None:
            return csv_bytes
        if isinstance(output, (str, Path)):
            Path(output).write_bytes(csv_bytes)
            print(f"[matrix] wrote {output}")
        else:
            output.write(buf.getvalue())
        return None

    def __str__(self) -> str:
        return self.to_text()
