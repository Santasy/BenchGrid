"""
Unified plan model for BenchGrid.

A WorkPlan is data: workdir, program, how grid records join the command line,
a grid of combinations and an execution policy. Rendering never executes
anything; ``WorkPlan.run`` is the dry-run / execute seam that puts work on the
shared PoolRunner.

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

Resource classification is deliberately kept out of this module. Callers that
know a command's CPU or memory cost can pass an explicit ``threads`` value to
the execution seam.
"""

import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .runner import PoolRunner
from .utils import Grid, WorkRecord, expand_records


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
    """One shell invocation: 1-indexed id, command, tags, and working dir."""

    id: int
    command: str
    tags: Mapping[str, str] = field(default_factory=dict)
    cwd: str | Path | None = None

    def execute(self) -> tuple[int, float]:
        """Shell out to ``self.command``; returns (exit code, elapsed)."""
        start = time.perf_counter()
        rc = subprocess.call(self.command, shell=True, cwd=self.cwd)
        return rc, time.perf_counter() - start


@dataclass
class WorkPlan:
    """
    Declarative job over a grid of combinations.

    ``workdir`` is where each command runs. It is stored on every
    :class:`WorkCommand` rather than rendered into a shell ``cd`` prefix.
    ``program`` is the build script or already-built executable, ``args_join``
    how records map to the command line, ``fixed`` the positional arguments
    identical across every POSITIONAL command, and ``grid`` the varying
    parameters. A grid mapping is a Cartesian product; a list is verbatim and
    may contain :class:`WorkRecord` values carrying non-rendered tags.

    ``parallel`` is the declared concurrency policy: False pins execution to a
    single worker, True lets the pool spread commands across threads.
    ``single_core`` records the expectation that each command uses one CPU
    core. With ``iteration=FREEZE_DIMENSION``, ``freeze_key`` names the
    dimension whose values are exhausted one at a time.
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

    @property
    def cwd(self) -> Path | None:
        value = str(self.workdir).rstrip("/")
        return Path(value) if value else None

    def build_command(self, record: Mapping[str, str]) -> str:
        """
        Compose one shell invocation from a grid record.

        ``ASSIGNMENT``: ``<program> NAME=VALUE ...``;
        ``POSITIONAL``: ``<program> <fixed...> <values...>``.
        Empty-valued arguments are dropped; values follow record key order.
        """
        if self.args_join is ArgsJoinPolicy.POSITIONAL:
            parts = [*self.fixed, *record.values()]
        else:
            parts = [f"{name}={value}" for name, value in record.items() if value != ""]
        parts = [part for part in parts if part != ""]
        return f"{self.program} {' '.join(parts)}".strip()

    def build_commands(self, id_offset: int = 1) -> list[WorkCommand]:
        """Render every grid combination into one WorkCommand."""
        commands: list[WorkCommand] = []
        for offset, record in enumerate(expand_records(self.grid)):
            tags = dict(record.values)
            tags.update(record.tags)
            commands.append(
                WorkCommand(
                    offset + id_offset,
                    self.build_command(record.values),
                    tags,
                    self.cwd,
                )
            )
        return commands

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
        """Render and execute this plan through the shared execution seam."""
        return _execute_commands(
            self.build_commands(),
            label=self.args_join.value,
            dry_run=dry_run,
            verbose=verbose,
            threads=threads,
            parallel=self.parallel,
            iteration=self.iteration,
            freeze_key=self.freeze_key,
        )


def execute_commands(
    commands: Sequence[WorkCommand],
    *,
    label: str = "pool",
    dry_run: bool = False,
    verbose: bool = False,
    threads: int | None = None,
    parallel: bool = True,
    iteration: IterationPolicy = IterationPolicy.ALL_AT_ONCE,
    freeze_key: str | None = None,
) -> int:
    """Execute already-rendered commands with the shared pool semantics."""
    return _execute_commands(
        commands,
        label=label,
        dry_run=dry_run,
        verbose=verbose,
        threads=threads,
        parallel=parallel,
        iteration=iteration,
        freeze_key=freeze_key,
    )


def _execute_commands(
    commands: Sequence[WorkCommand],
    *,
    label: str,
    dry_run: bool,
    verbose: bool,
    threads: int | None,
    parallel: bool,
    iteration: IterationPolicy,
    freeze_key: str | None,
) -> int:
    rendered = list(commands)
    if verbose or dry_run:
        for command in rendered:
            print(f"[{label}] {command.id:3d}: {command.command}", flush=True)
    if dry_run:
        print(f"\n[{label}] Dry run: {len(rendered)} command(s), not executing.")
        return 0
    if not rendered:
        print(f"\n[{label}] Nothing to run.")
        return 0

    if not parallel:
        threads = 1

    def _execute_batch(batch: list[WorkCommand]) -> None:
        def _report_completion(index: int, outcome: tuple[int, float]) -> None:
            _rc, elapsed = outcome
            print(
                f"[{batch[index].id:3d}] Elapsed [secs]: {elapsed:.3f}",
                flush=True,
            )

        for command in batch:
            print(f"[{command.id:3d}] Running {command.command}", flush=True)
        outcomes.append(
            PoolRunner(WorkCommand.execute, threads=threads).execute(
                batch, on_complete=_report_completion
            )
        )

    outcomes: list[list[tuple[int, float]]] = []
    if iteration is IterationPolicy.FREEZE_DIMENSION:
        for key, group in freeze_groups(rendered, freeze_key):
            print(
                f"\n===== {freeze_key}={key}: {len(group)} command(s) =====",
                flush=True,
            )
            _execute_batch(group)
    else:
        _execute_batch(rendered)

    failed = [rc for batch in outcomes for rc, _ in batch if rc != 0]
    if failed:
        print(f"\n[{label}] {len(failed)} command(s) failed.")
        return 1
    print("\n===== Finished =====")
    return 0


def freeze_groups(
    commands: Sequence[WorkCommand], freeze_key: str | None
) -> list[tuple[str, list[WorkCommand]]]:
    """
    Split commands by the frozen dimension, keeping first-appearance order.

    Grouping is by tag over the whole batch, not by consecutive runs, so a
    merged pool of per-tree plans still executes every seed=1 command across
    all trees before any seed=2 command starts.
    """
    ordered: dict[str, list[WorkCommand]] = {}
    for command in commands:
        key = command.tags.get(freeze_key or "", "")
        ordered.setdefault(key, []).append(command)
    return list(ordered.items())


__all__ = [
    "ArgsJoinPolicy",
    "IterationPolicy",
    "WorkCommand",
    "WorkPlan",
    "execute_commands",
    "freeze_groups",
]
