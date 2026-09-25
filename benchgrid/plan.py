"""
Unified plan model for BenchGrid.

A WorkPlan is data: workdir, program, how grid records join the command line,
a grid of combinations and an execution policy. Rendering never executes
anything; ``WorkPlan.run`` is the dry-run / execute seam that puts work on
threads through the shared PoolRunner.

Two join policies cover both feature surfaces:

- ``ASSIGNMENT`` renders each grid record as ``NAME=VALUE`` pairs (the
  construction surface: build a binary per combination).
- ``POSITIONAL`` renders ``<fixed args> <grid values in record key order>``
  (the benchmark surface: invoke an already-built binary per combination).

Benchmark semantics: each rendered command is one invocation that typically
occupies a single CPU core (``single_core=True`` is the default). Parallel
execution is therefore many independent single-core processes, never threading
inside a task.

Iteration policy controls how those processes are scheduled:
all at once, or frozen one dimension at a time (e.g. exhaust every combination
at seed=1, then seed=2, ...).

Resource classification (isolating "expensive" vs. regular commands and
deriving per-class thread limits such as ``available_GB / expensive_RAM_use``)
is a documented future feature, intentionally not implemented here: per-class
RAM use depends on run parameters, so it belongs one layer (or more) above
this module.
"""

import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .runner import PoolRunner
from .utils import Grid, expand_grid


class ArgsJoinPolicy(Enum):
    """How a grid record becomes the command line after the program."""

    ASSIGNMENT = "assignment"  # program NAME=VALUE ... (builds)
    POSITIONAL = "positional"  # program <fixed...> <grid values...> (runs)


class IterationPolicy(Enum):
    """How a plan schedules its rendered commands over the pool."""

    ALL_AT_ONCE = "all_at_once"
    FREEZE_DIMENSION = "freeze_dimension"


@dataclass(frozen=True)
class WorkCommand:
    """One shell invocation: 1-indexed id + the command string.

    ``tags`` carries the grid record (dimension name → value) for that command,
    so a freeze-dimension executor can group commands by their frozen value.
    """

    id: int
    command: str
    tags: Mapping[str, str] = field(default_factory=dict)

    def execute(self) -> tuple[int, float]:
        """Shell out to ``self.command``; returns (exit code, elapsed)."""
        start = time.perf_counter()
        rc = subprocess.call(self.command, shell=True)
        return rc, time.perf_counter() - start


@dataclass
class WorkPlan:
    """
    Declarative job over a grid of combinations.

    ``workdir`` is where each command changes into (omitted from the rendered
    string when empty), ``program`` the build script or already-built
    executable, ``args_join`` how records map to the command line, ``fixed``
    the positional arguments identical across every POSITIONAL command (e.g.
    version name, storage folder, config file) and ``grid`` the varying
    parameters (dict = Cartesian product, list of records = verbatim).

    ``parallel`` is the declared concurrency policy: False pins execution to a
    single worker (scripts that reconfigure one shared build directory, e.g.
    build_here.sh), True lets the pool spread commands across threads.
    ``single_core`` records the expectation that each command uses one CPU
    core; it is informational today and reserved for a future
    resource-classification feature. With ``iteration=FREEZE_DIMENSION``,
    ``freeze_key`` names the dimension whose values are exhausted one at a
    time (e.g. SEED).
    """

    workdir: str | Path
    program: str
    args_join: ArgsJoinPolicy
    fixed: Sequence[str] = field(default=(), kw_only=True)
    grid: Grid = field(default=None, kw_only=True)
    parallel: bool = field(default=True, kw_only=True)
    single_core: bool = field(default=True, kw_only=True)
    iteration: IterationPolicy = field(
        default=IterationPolicy.ALL_AT_ONCE, kw_only=True
    )
    freeze_key: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.iteration is IterationPolicy.FREEZE_DIMENSION and not self.freeze_key:
            raise ValueError(
                "FREEZE_DIMENSION iteration requires freeze_key (e.g. 'SEED')"
            )
        if self.args_join is ArgsJoinPolicy.ASSIGNMENT and self.fixed:
            raise ValueError("'fixed' is a POSITIONAL-join concept")

    def build_command(self, record: Mapping[str, str]) -> str:
        """
        Compose one shell invocation from a grid record.

        ``ASSIGNMENT``: ``cd <workdir>/ && <program> NAME=VALUE ...``;
        ``POSITIONAL``: ``cd <workdir>/ && <program> <fixed...> <values...>``.
        The ``cd`` prefix is omitted for an empty workdir; empty-valued
        arguments are dropped; values follow the record's key order.
        """
        wd = str(self.workdir).rstrip("/")
        prefix = f"cd {wd}/ && " if wd else ""
        if self.args_join is ArgsJoinPolicy.POSITIONAL:
            parts = [*self.fixed, *record.values()]
        else:
            parts = [f"{name}={value}" for name, value in record.items() if value != ""]
        parts = [part for part in parts if part != ""]
        return f"{prefix}{self.program} {' '.join(parts)}".strip()

    def build_commands(self, id_offset: int = 1) -> list[WorkCommand]:
        """Render every grid combination into one WorkCommand."""
        return [
            WorkCommand(idx, self.build_command(record), record)
            for idx, record in enumerate(expand_grid(self.grid), start=id_offset)
        ]

    def print_commands(self, prefix: str = "[work]") -> None:
        """Dry-run preview: one line per command, nothing is executed."""
        for cmd in self.build_commands():
            print(f"{prefix} {cmd.id:3d}: {cmd.command}", flush=True)

    def run(
        self,
        *,
        dry_run: bool = False,
        verbose: bool = False,
        threads: int | None = None,
    ) -> int:
        """
        Orchestrator seam: render, then print or execute across the pool.

        ``dry_run`` renders and prints the plan without touching the filesystem
        and is mutually exclusive with execution. A non-parallel plan always
        runs with a single worker regardless of ``threads``. With
        ``iteration=FREEZE_DIMENSION``, commands are exhausted one
        frozen-dimension value at a time (e.g. all seed=1 combinations, then
        seed=2, ...), each value its own pool batch.

        Returns:
            Exit code: 0 success / nothing to do, 1 on any failing command.
        """
        commands = self.build_commands()
        label = self.args_join.value
        if verbose or dry_run:
            self.print_commands(prefix=f"[{label}]")
        if dry_run:
            print(f"\n[{label}] Dry run: {len(commands)} command(s), not executing.")
            return 0
        if not commands:
            print(f"\n[{label}] Nothing to run.")
            return 0

        if not self.parallel:
            threads = 1

        def _report_completion(index: int, outcome: tuple[int, float]) -> None:
            _rc, elapsed = outcome
            print(
                f"[{commands[index].id:3d}] Elapsed [secs]: {elapsed:.3f}",
                flush=True,
            )

        def _execute_batch(batch: list[WorkCommand]) -> None:
            for cmd in batch:
                print(f"[{cmd.id:3d}] Running {cmd.command}", flush=True)
            outcomes.append(
                PoolRunner(WorkCommand.execute, threads=threads).execute(
                    batch, on_complete=_report_completion
                )
            )

        outcomes: list[list[tuple[int, float]]] = []
        if self.iteration is IterationPolicy.FREEZE_DIMENSION:
            for key, group in _freeze_groups(commands, self.freeze_key):
                print(
                    f"\n===== {self.freeze_key}={key}: {len(group)} command(s) =====",
                    flush=True,
                )
                _execute_batch(group)
        else:
            _execute_batch(commands)

        failed = [rc for batch in outcomes for rc, _ in batch if rc != 0]
        if failed:
            print(f"\n[{label}] {len(failed)} command(s) failed.")
            return 1
        print("\n===== Finished =====")
        return 0


def _freeze_groups(
    commands: list[WorkCommand], freeze_key: str | None
) -> list[tuple[str, list[WorkCommand]]]:
    """Split commands by the frozen dimension, keeping first-appearance order."""
    ordered: dict[str, list[WorkCommand]] = {}
    for cmd in commands:
        key = cmd.tags.get(freeze_key or "", "")
        ordered.setdefault(key, []).append(cmd)
    return [(key, group) for key, group in ordered.items()]


__all__ = [
    "ArgsJoinPolicy",
    "IterationPolicy",
    "WorkCommand",
    "WorkPlan",
]
