"""Flat command pools that share one worker budget across many plans."""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from .plan import IterationPolicy, WorkCommand, WorkPlan, execute_commands
from .resources import resolve_thread_count


@dataclass
class CommandPool:
    """
    A flat list of rendered commands that share one worker budget.

    This is the scheduling surface for callers that build many small plans
    (one per build variant, workload or tree) but want them to run together
    instead of each plan paying for its own pool. Command ids are renumbered
    from 1 in pool order, so merged plans stay individually addressable while
    the pool keeps one stable numbering.
    """

    name: str
    commands: Sequence[WorkCommand] = field(default_factory=tuple)
    threads: int | None = None
    parallel: bool = True
    iteration: IterationPolicy = IterationPolicy.ALL_AT_ONCE
    freeze_key: str | None = None
    cpu_reserve: int = 1

    def __post_init__(self) -> None:
        self.commands = [
            replace(command, id=index)
            for index, command in enumerate(self.commands, start=1)
        ]

    def resolved_threads(self) -> int:
        return resolve_thread_count(
            self.threads,
            cpu_reserve=self.cpu_reserve,
            command_count=len(self.commands),
        )

    def run(
        self, *, dry_run: bool = False, verbose: bool = False
    ) -> int:
        return execute_commands(
            self.commands,
            label=self.name,
            dry_run=dry_run,
            verbose=verbose,
            threads=self.resolved_threads(),
            parallel=self.parallel,
            iteration=self.iteration,
            freeze_key=self.freeze_key,
        )


def merge_plans(
    plans: Sequence[WorkPlan],
    name: str,
    *,
    threads: int | None = None,
    parallel: bool = True,
    iteration: IterationPolicy = IterationPolicy.ALL_AT_ONCE,
    freeze_key: str | None = None,
    cpu_reserve: int = 1,
) -> CommandPool:
    """Render several plans and schedule the union in one pool."""
    commands: list[WorkCommand] = []
    for plan in plans:
        commands.extend(plan.build_commands())
    return CommandPool(
        name=name,
        commands=commands,
        threads=threads,
        parallel=parallel,
        iteration=iteration,
        freeze_key=freeze_key,
        cpu_reserve=cpu_reserve,
    )


__all__ = ["CommandPool", "merge_plans"]
