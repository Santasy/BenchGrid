"""Render and validate workload-agnostic work plans without executing them."""

from pathlib import Path

from benchgrid import (
    ArgsJoinPolicy,
    CommandPool,
    IterationPolicy,
    WorkCommand,
    WorkPlan,
    WorkRecord,
    available_cpus,
    expand_records,
    merge_plans,
    resolve_thread_count,
)
from benchgrid.utils import expand_grid

first_plan = WorkPlan(
    workdir="project",
    program="run",
    args_join=ArgsJoinPolicy.ASSIGNMENT,
    grid={"PARAM_A": ["1", "2"], "PARAM_B": ["x", "y"]},
)
assert first_plan.parallel is True
commands = first_plan.build_commands()
assert [command.id for command in commands] == [1, 2, 3, 4]
assert [command.command for command in commands] == [
    "run PARAM_A=1 PARAM_B=x",
    "run PARAM_A=1 PARAM_B=y",
    "run PARAM_A=2 PARAM_B=x",
    "run PARAM_A=2 PARAM_B=y",
]
assert [str(command.cwd) for command in commands] == ["project"] * 4
assert commands[0].tags == {"PARAM_A": "1", "PARAM_B": "x"}

plan_undefined_optional = WorkPlan(
    workdir="",
    program="run",
    args_join=ArgsJoinPolicy.ASSIGNMENT,
    grid={"PARAM_A": ["1"], "OPTIONAL": [""]},
).build_commands()
assert plan_undefined_optional[0].command == "run PARAM_A=1"
assert plan_undefined_optional[0].cwd is None

solo_plan = WorkPlan("project", "run", ArgsJoinPolicy.ASSIGNMENT).build_commands()
assert solo_plan == [WorkCommand(1, "run", cwd=Path("project"))]

try:
    WorkPlan("project", "run", ArgsJoinPolicy.ASSIGNMENT, fixed=("config",))
except ValueError:
    pass
else:
    raise AssertionError("ASSIGNMENT join must reject fixed arguments")

grid = expand_grid({"PARAM_A": ["1", "2"], "PARAM_B": ["x", "y"]})
assert len(grid) == 4 and grid[0] == {"PARAM_A": "1", "PARAM_B": "x"}

records = expand_records(
    [
        WorkRecord({"ARGS": "1"}, {"SEED": "1"}),
        WorkRecord({"ARGS": "2"}, {"SEED": "1"}),
        WorkRecord({"ARGS": "1"}, {"SEED": "2"}),
    ]
)
assert records[0].tags == {"SEED": "1"}
assert records[0].values == {"ARGS": "1"}

seeded = WorkPlan(
    workdir="project",
    program="run",
    args_join=ArgsJoinPolicy.POSITIONAL,
    fixed=["--workload", "a"],
    grid=[
        WorkRecord({"SEED": "1", "ARGS": "x"}, {"SEED": "1"}),
        WorkRecord({"SEED": "1", "ARGS": "y"}, {"SEED": "1"}),
        WorkRecord({"SEED": "2", "ARGS": "x"}, {"SEED": "2"}),
    ],
    iteration=IterationPolicy.FREEZE_DIMENSION,
    freeze_key="SEED",
)
seeded_commands = seeded.build_commands()
assert [command.command for command in seeded_commands] == [
    "run --workload a 1 x",
    "run --workload a 1 y",
    "run --workload a 2 x",
]
assert [command.tags["SEED"] for command in seeded_commands] == ["1", "1", "2"]

try:
    WorkPlan(
        workdir="project",
        program="run",
        args_join=ArgsJoinPolicy.POSITIONAL,
        iteration=IterationPolicy.FREEZE_DIMENSION,
    )
except ValueError:
    pass
else:
    raise AssertionError("FREEZE_DIMENSION iteration must require freeze_key")

merged = merge_plans([first_plan, seeded], name="all")
assert isinstance(merged, CommandPool)
assert len(merged.commands) == 7
assert [command.id for command in merged.commands] == list(range(1, 8))
assert merged.resolved_threads() == resolve_thread_count(
    None, cpu_reserve=merged.cpu_reserve, command_count=7
)
assert merged.resolved_threads() <= available_cpus(1)
assert resolve_thread_count(1_000, cpu_reserve=1) == available_cpus(1)
assert resolve_thread_count(None, cpu_reserve=0) <= available_cpus(0)
assert resolve_thread_count(None, cpu_reserve=1, command_count=2) == 2

frozen = merge_plans(
    [seeded],
    name="frozen",
    iteration=IterationPolicy.FREEZE_DIMENSION,
    freeze_key="SEED",
)
assert frozen.run(dry_run=True) == 0

print("ok")
