"""Render and validate a workload-agnostic work plan without executing it."""

from benchgrid import ArgsJoinPolicy, WorkCommand, WorkPlan
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
    "cd project/ && run PARAM_A=1 PARAM_B=x",
    "cd project/ && run PARAM_A=1 PARAM_B=y",
    "cd project/ && run PARAM_A=2 PARAM_B=x",
    "cd project/ && run PARAM_A=2 PARAM_B=y",
]

plan_undefined_optional = WorkPlan(
    workdir="",
    program="run",
    args_join=ArgsJoinPolicy.ASSIGNMENT,
    grid={"PARAM_A": ["1"], "OPTIONAL": [""]},
).build_commands()
assert plan_undefined_optional[0].command == "run PARAM_A=1"

solo_plan = WorkPlan("project", "run", ArgsJoinPolicy.ASSIGNMENT).build_commands()
assert solo_plan == [WorkCommand(1, "cd project/ && run")]

try:
    WorkPlan("project", "run", ArgsJoinPolicy.ASSIGNMENT, fixed=("config",))
except ValueError:
    pass
else:
    raise AssertionError("ASSIGNMENT join must reject fixed arguments")

grid = expand_grid({"PARAM_A": ["1", "2"], "PARAM_B": ["x", "y"]})
assert len(grid) == 4 and grid[0] == {"PARAM_A": "1", "PARAM_B": "x"}
print("ok")
