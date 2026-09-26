"""
Grid input model shared by every feature that maps combinations to commands.

A Grid is either a dict of name → value collections (expanded as a Cartesian
product) or a pre-filtered list of records (one command per record). A record
can also carry tags that do not appear on the rendered command line, which is
useful for grouping or freezing a dimension such as ``SEED``.
"""

import itertools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class WorkRecord:
    """One rendered grid record plus metadata used only for scheduling."""

    values: Mapping[str, str]
    tags: Mapping[str, str] = field(default_factory=dict)

    def normalized(self) -> "WorkRecord":
        return WorkRecord(dict(self.values), dict(self.tags))


#: Grid input: a mapping of names to value collections (Cartesian product) or
#: a pre-filtered list of records (exactly one record per combination).
Grid = (
    Mapping[str, Iterable[str]]
    | list[Mapping[str, str] | WorkRecord]
    | None
)


def expand_records(grid: Grid) -> list[WorkRecord]:
    """Expand a grid into records while preserving non-rendered tags."""
    if grid is None:
        return [WorkRecord({})]
    if isinstance(grid, list):
        records: list[WorkRecord] = []
        for record in grid:
            if isinstance(record, WorkRecord):
                records.append(record.normalized())
            elif isinstance(record, Mapping):
                records.append(WorkRecord(dict(record)))
            else:
                raise TypeError(
                    "grid records must be mappings or WorkRecord instances, "
                    f"got {type(record).__name__}"
                )
        return records
    if not isinstance(grid, Mapping):
        raise TypeError(
            f"grid must be a dict or a list of records, got {type(grid).__name__}"
        )
    if not grid:
        return [WorkRecord({})]

    names = list(grid)
    collections: list[Iterable[str]] = [
        (values,) if isinstance(values, str) else values
        for values in grid.values()
    ]
    return [
        WorkRecord(dict(zip(names, combination)))
        for combination in itertools.product(*collections)
    ]


def expand_grid(grid: Grid) -> list[dict[str, str]]:
    """Return only the rendered values from :func:`expand_records`."""
    return [dict(record.values) for record in expand_records(grid)]


__all__ = ["Grid", "WorkRecord", "expand_grid", "expand_records"]
