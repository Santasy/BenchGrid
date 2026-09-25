"""
Shared multiprocess execution engine for BenchGrid.

``PoolRunner`` is what the unified plan model uses to execute a batch of
commands built from a grid: it decides how many worker processes to spawn
(``threads`` or one fewer than the logical CPUs, clamped to the batch size) and
keeps track of how many commands are running/collected. Completion is surfaced
as it happens: an optional ``on_complete`` callback receives each result the
moment its task finishes (in completion order), while ``execute`` still returns
results in input order. Nothing here knows about building or running
specifically — the worker callable is injected.
"""

import multiprocessing
from threading import Event
from typing import Generic, TypeVar
from collections.abc import Callable, Sequence

from psutil import cpu_count

ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT")


def default_threads() -> int:
    """Parallel-worker default: one fewer than the logical CPUs, minimum one."""
    cpus = cpu_count() or 1
    return max(1, cpus - 1)


def mp_context() -> multiprocessing.context.BaseContext:
    """
    Multiprocessing context for worker pools.

    Prefers ``fork``: since Python 3.14 the default start method is forkserver,
    which re-imports the main module and breaks pools launched from stdin or
    from top-level scripts (e.g. the ef-test resources). Falls back to the
    platform default where fork is unavailable.
    """
    try:
        return multiprocessing.get_context("fork")
    except ValueError:
        return multiprocessing.get_context()


class PoolRunner(Generic[ItemT, ResultT]):
    """
    Multiprocess executor that applies a worker callable to a batch of items.

    Principal responsibility: decide how many worker processes to use and track
    how the batch is progressing (``threads`` / ``running`` / ``finished``).
    Tasks are submitted with ``apply_async`` all at once; a per-task callback
    fires the instant each worker finishes, so ``on_complete`` can observe
    results in completion order without waiting for the whole batch.
    """

    def __init__(
        self,
        worker: Callable[[ItemT], ResultT],
        *,
        threads: int | None = None,
        worker_init: Callable[..., object] | None = None,
        worker_init_args: Sequence[object] = (),
    ) -> None:
        self._worker = worker
        self._requested = default_threads() if threads is None else threads
        self._worker_init = worker_init
        self._worker_init_args = tuple(worker_init_args)
        self._threads = 0
        self._total = 0
        self._finished = 0

    @property
    def threads(self) -> int:
        """Worker-process count used by the last ``execute`` batch."""
        return self._threads

    @property
    def running(self) -> int:
        """Commands submitted but not yet collected by ``execute``."""
        return max(0, self._total - self._finished)

    @property
    def finished(self) -> int:
        """Collected results from the last ``execute`` batch."""
        return self._finished

    def execute(
        self,
        items: Sequence[ItemT],
        *,
        on_complete: Callable[[int, ResultT], None] | None = None,
    ) -> list[ResultT]:
        """
        Run ``worker`` over every item and return results in input order.

        All tasks are submitted up front via ``apply_async``; the moment a
        worker finishes, a per-task callback places its result and, when
        ``on_complete`` is given, invokes it with ``(index, result)`` — the
        0-based position in ``items`` and the worker's return value — so the
        callback observes completion order, not submission order. Callbacks run
        on the pool's internal result-handler thread and must be cheap (enqueue
        or bookkeeping only). A worker exception is captured and re-raised from
        here after the batch has drained.

        ``threads`` (or the CPU-derived default) is clamped to the batch size;
        an empty batch executes nothing and creates no pool.
        """
        n = len(items)
        if not n:
            return []
        self._threads = max(1, min(self._requested, n))
        self._total = n
        self._finished = 0
        print(
            f"\n===== Running {n} command(s) with {self._threads} workers =====",
            flush=True,
        )

        pool = mp_context().Pool(
            processes=self._threads,
            initializer=self._worker_init,
            initargs=self._worker_init_args,
        )

        collected: dict[int, ResultT] = {}
        failure: list[BaseException] = []
        done = Event()

        def _finish(index: int, result: ResultT) -> None:
            collected[index] = result
            self._finished += 1
            if on_complete is not None:
                on_complete(index, result)
            if self._finished == n:
                done.set()

        def _fail(index: int, exc: BaseException) -> None:
            failure.append(exc)
            done.set()

        try:
            for index, item in enumerate(items):
                pool.apply_async(
                    self._worker,
                    (item,),
                    callback=lambda res, i=index: _finish(i, res),
                    error_callback=lambda exc, i=index: _fail(i, exc),
                )
            done.wait()
        finally:
            pool.close()
            pool.join()
            self._total = 0

        if failure:
            raise failure[0]
        return [collected[i] for i in range(n)]


__all__ = ["PoolRunner", "default_threads", "mp_context"]
