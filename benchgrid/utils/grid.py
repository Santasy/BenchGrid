"""
Grid input model shared by every feature that maps combinations to commands.

A Grid is either a dict of name → value collections (expanded as a Cartesian
product) or a pre-filtered list of assignment dicts (one command per record).
Only ``expand_grid`` interprets it; nothing here executes anything.
"""

import itertools
from collections.abc import Iterable, Mapping

#: Grid input: a mapping of names to value collections (Cartesian product) or
#: a pre-filtered list of assignment dicts (exactly one record per combination).
Grid = Mapping[str, Iterable[str]] | list[dict[str, str]] | None


def expand_grid(grid: Grid) -> list[dict[str, str]]:
    """
    Flatten a grid into one assignment dict per command.

    - ``None`` / ``{}`` → a single empty dict (bare program invocation).
    - dict → Cartesian product over the value collections; a plain string
      value counts as a single-element collection; an empty collection yields
      no commands.
    - list[dict] → copied verbatim, one command per record (the pre-filtered
      form, e.g. insert_only's validity-pruned combinations).
    """
    if grid is None:
        return [{}]
    if isinstance(grid, list):
        return [dict(record) for record in grid]
    if not isinstance(grid, Mapping):
        raise TypeError(
            f"grid must be a dict or a list of dicts, got {type(grid).__name__}"
        )
    if not grid:
        return [{}]

    names = list(grid)
    collections: list[Iterable[str]] = [
        (values,) if isinstance(values, str) else values
        for values in grid.values()
    ]
    return [dict(zip(names, combination)) for combination in itertools.product(*collections)]


__all__ = ["Grid", "expand_grid"]