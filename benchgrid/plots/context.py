"""
The context a tab body receives — plain data, no streamlit.

A project-declared tab is a function of this context: the adapter, the loaded
dataset and the current sidebar/tab selection.  Keeping it streamlit-free means
a headless caller (a CLI, a test) can build one and drive the project tab's
frame logic without a UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import polars as pl

from ..data import _filter_frame, final_rows

if TYPE_CHECKING:
    from ..data import Dataset
    from .adapter import PlotsAdapter

__all__ = ["TabContext"]


@dataclass
class TabContext:
    """Adapter + dataset + the sidebar's current filters."""

    adapter: PlotsAdapter
    dataset: Dataset
    #: Sidebar dimension selections (exact-match, string-compared).
    filters: dict[str, Any] = field(default_factory=dict)
    #: Sidebar aggregation over the repeated measures ("mean" by default).
    agg: str = "mean"
    #: Sidebar x keys, when the project shows a compare-x option.
    x_keys: list[str] = field(default_factory=list)
    #: Tab-local step (batch phase) selection; ``None`` = unfiltered.  Written
    #: by :func:`~benchgrid.plots.app.select_step`.
    step: Any = None

    def frame(self, source: str, *, final: bool = False) -> pl.DataFrame:
        """
        The *source* frame with the sidebar filters and *step* applied.

        With *final*, keep only the last measure per run — the maximum
        *adapter.progress_key* within *adapter.run_keys* (the shared last-of
        rule, :func:`~benchgrid.data.final_rows`) — i.e. the comparable
        per-run value of the selected step for cumulative samplers.  The step
        is applied *first*, so *final* is that step's final measure (for
        insert_only: the greatest ``n`` of the selected step).
        """
        df = _filter_frame(self.dataset.frame(source), self.filters)
        step_key = self.adapter.step_key
        if self.step is not None and step_key:
            df = df.filter(pl.col(step_key) == self.step)
        if final and self.adapter.progress_key:
            df = final_rows(df, self.adapter.run_keys, self.adapter.progress_key)
        return df
