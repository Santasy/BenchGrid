"""
BenchGrid — generic, workload-agnostic tooling that keeps the
compilation → execution → data → analysis chain consistent for any project.

The project supplies grids/schemas/params; BenchGrid supplies the generic
engine. ``runner`` is the shared multiprocess executor; ``plan`` is the unified
plan model whose ``args_join`` policy covers both the build surface
(``ArgsJoinPolicy.ASSIGNMENT``) and the benchmark surface
(``ArgsJoinPolicy.POSITIONAL``).

The analysis side (schema-driven load and aggregation over Polars, common
output like terminal/tables/csv, and streamlit plotting) is absorbed from
benchplots — which stays available untouched — and is provided behind the
package's own surface once the chain layers are folded in.
"""

__version__ = "0.2.0"

# === Shared engine ===
from . import utils
from .runner import PoolRunner, default_threads, mp_context

# === Unified plan model (build = ASSIGNMENT join, run = POSITIONAL join) ===
from .plan import ArgsJoinPolicy, IterationPolicy, WorkCommand, WorkPlan

__all__ = [
    "__version__",
    "utils",
    "ArgsJoinPolicy",
    "IterationPolicy",
    "WorkCommand",
    "WorkPlan",
    "PoolRunner",
    "default_threads",
    "mp_context",
]
