"""
BenchGrid: one place to express "render N commands, then run N commands".

Nothing in BenchGrid knows what a build script, a benchmark binary or a
workload *means*. Callers describe programs, grid dimensions and execution
policy as data (:class:`benchgrid.plan.WorkPlan`), and BenchGrid owns the
mechanics: expanding the grid, rendering command lines, tracking which
combination a command belongs to, and scheduling the work on one shared
thread pool with a CPU budget.

A typical pipeline looks like::

    stage_a = WorkPlan(
        workdir=project,
        program="python3 build.py",
        args_join=ArgsJoinPolicy.POSITIONAL,
        fixed=["--name", "build"],
        grid={"SIZE": ["32", "64"]},
    )
    stage_b = WorkPlan(
        workdir=project,
        program="./bench --workload a",
        args_join=ArgsJoinPolicy.POSITIONAL,
        fixed=["--workload", "a"],
        grid={"SEED": ["1", "2"]},
    )
    exit_code = merge_plans([stage_a, stage_b], name="all").run()

Because plans are plain data they can be inspected, serialised, asserted
against and merged before a single process is spawned.
"""

__version__ = "1.2.0"

# === Shared engine ===
from .runner import PoolRunner, default_threads, mp_context
from .utils import Grid, WorkRecord, expand_grid, expand_records

from .plan import (
    ArgsJoinPolicy,
    IterationPolicy,
    WorkCommand,
    WorkPlan,
    execute_commands,
    freeze_groups,
)
from .pool import CommandPool, merge_plans
from .resources import available_cpus, resolve_thread_count

# === Data / analysis layer (schema-driven Polars scan + query) ===
from . import data, scan, schema
from .data import Dataset, Matrix
from .scan import scan_csv, scan_csv_lazy, scan_json
from .schema import (
    CAST_TYPE_MAP,
    FilePattern,
    FolderLevel,
    LineSchema,
    ScanSchema,
    read_schemas,
)

__all__ = [
    "ArgsJoinPolicy",
    "CommandPool",
    "Grid",
    "IterationPolicy",
    "PoolRunner",
    "WorkCommand",
    "WorkPlan",
    "WorkRecord",
    "available_cpus",
    "execute_commands",
    "expand_grid",
    "freeze_groups",
    "expand_records",
    "merge_plans",
    "resolve_thread_count",
    "PoolRunner",
    "default_threads",
    "mp_context",
    "schema",
    "scan",
    "data",
    "FolderLevel",
    "FilePattern",
    "LineSchema",
    "ScanSchema",
    "read_schemas",
    "CAST_TYPE_MAP",
    "scan_csv",
    "scan_csv_lazy",
    "scan_json",
    "Dataset",
    "Matrix",
]
