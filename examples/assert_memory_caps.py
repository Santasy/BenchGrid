"""Self-check the per-command memory cap and the pool memory budget.

    venv/bin/python scripts/tools/BenchGrid/examples/assert_memory_caps.py

Three claims, each asserted with real subprocesses:

1. ``mem_limit`` (RLIMIT_AS): a command allocating 512 MiB dies under a
   64 MiB cap and succeeds without one.
2. ``memory_budget``: two commands whose estimates exactly fill the budget
   may not overlap (admission is serialized).
3. A command estimating more than the whole budget is rejected up front —
   it could never start.
"""

import sys
import time

from benchgrid import WorkCommand
from benchgrid.plan import execute_commands

GB = 2**30

_ALLOC = (
    f'{sys.executable!r} -c "b = bytearray(512 * 2**20); print(len(b))"'
)
_SLEEP = f'{sys.executable!r} -c "import time; time.sleep(0.4)"'


def test_mem_limit() -> None:
    capped = WorkCommand(1, _ALLOC, mem_limit=64 * 2**20)
    rc, _ = capped.execute()
    assert rc != 0, "512 MiB allocation survived a 64 MiB RLIMIT_AS"
    free = WorkCommand(1, _ALLOC)
    rc, _ = free.execute()
    assert rc == 0, "unbounded command failed"


def test_memory_budget() -> None:
    commands = [
        WorkCommand(1, _SLEEP, mem_estimate=GB),
        WorkCommand(2, _SLEEP, mem_estimate=GB),
    ]
    started = time.perf_counter()
    rc = execute_commands(
        commands, threads=2, memory_budget=GB, label="budget-check"
    )
    elapsed = time.perf_counter() - started
    assert rc == 0
    assert elapsed >= 0.75, f"two 1-GiB-estimated commands overlapped ({elapsed:.2f}s)"


def test_impossible_estimate() -> None:
    commands = [WorkCommand(1, _SLEEP, mem_estimate=4 * GB)]
    try:
        execute_commands(commands, threads=1, memory_budget=GB, label="never")
    except ValueError as error:
        assert "exceeds memory_budget" in str(error)
        return
    raise AssertionError("estimate above the budget was accepted")


if __name__ == "__main__":
    test_mem_limit()
    test_memory_budget()
    test_impossible_estimate()
    print("memory caps + budget: OK")
