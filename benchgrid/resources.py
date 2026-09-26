"""Host CPU budgeting shared by every pool that needs one worker per core."""

import os


def available_cpus(cpu_reserve: int = 1) -> int:
    """Return usable cores after holding back ``cpu_reserve`` cores."""
    total = os.cpu_count() or 1
    return max(1, total - max(0, int(cpu_reserve)))


def resolve_thread_count(
    requested: int | None = None,
    *,
    cpu_reserve: int = 1,
    command_count: int | None = None,
) -> int:
    """
    Resolve a worker count from a request, the host CPU budget and the work.

    ``None`` requests the full budget. An explicit request is a ceiling that is
    clamped to the budget. The result is also clamped to the number of
    commands so empty pools and tiny pools never spawn idle workers.
    """
    budget = available_cpus(cpu_reserve)
    threads = budget if requested is None else min(int(requested), budget)
    if threads < 1:
        raise ValueError("requested thread count must be >= 1")
    if command_count is not None:
        threads = min(threads, max(1, command_count))
    return max(1, threads)
