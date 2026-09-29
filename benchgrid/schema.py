"""
Schema-driven reading model for BenchGrid.

A result tree is described entirely by a :class:`ScanSchema` — folder
hierarchy, file naming, and line format.
Schemas are declared **as JSON data** (:func:`read_schemas`),
readed from the file at load time
(the "schema segregates the tools" rule from the project knowledge docs).

Single source::

    {"pattern": "^(?P<version>[\\w.-]+)$", "name": "time", ...}

Multiple sources sharing the same folder tree are one JSON object of named
schemas, scanned together so the records they produce stay in the same key
space::

    {"time": {...}, "memory": {...}}

File layout have these primitives:

- :class:`FolderLevel` — one directory level; named groups become record keys.
- :class:`FilePattern` — matches file names; ``format="json"`` selects a JSON
  object stream read by :func:`benchgrid.scan_json`.
- :class:`LineSchema` — describes one data line; only used for
  ``format="lines"`` sources.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Supported ``FilePattern.format`` values.
FILE_FORMATS: tuple[str, ...] = ("lines", "json")

#: Accepted ``LineSchema.cast`` values (JSON strings; resolved at scan time).
CAST_TYPES: tuple[str, ...] = ("int", "float", "str")

#: Python type per ``LineSchema.cast`` name.
CAST_TYPE_MAP: dict[str, type] = {"int": int, "float": float, "str": str}


@dataclass
class FolderLevel:
    """
    One directory level of the expected folder hierarchy.

    Named groups in *pattern* become record keys; a pattern without named
    groups (e.g. ``r'^variants$'``) acts as a pass-through validator that
    matches the directory name without contributing keys.
    """

    pattern: str
    required: bool = True

    def compile(self) -> re.Pattern:
        return re.compile(self.pattern)

    def to_dict(self) -> dict[str, Any]:
        return {"pattern": self.pattern, "required": self.required}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FolderLevel":
        return cls(pattern=data["pattern"], required=bool(data.get("required", True)))


@dataclass
class FilePattern:
    """
    Matches file names inside the deepest folder level.

    Named groups become record keys.  *format* selects how the file body is
    parsed: ``"lines"`` (one record per line, via the scan schema's
    :class:`LineSchema`) or ``"json"`` (a JSON object stream; ``LineSchema``
    unused).  For ``"json"``, *object_marker* names the key that opens every
    object (e.g. ``"step"`` for ``{"step": ...}`` records); :func:`benchgrid.scan_json`
    uses it to split the comma-joined stream into one line per object.
    """

    pattern: str
    extensions: list[str] = field(default_factory=lambda: ["txt"])
    format: str = "lines"
    object_marker: str = ""

    def __post_init__(self) -> None:
        if self.format not in FILE_FORMATS:
            raise ValueError(
                f"Unknown file format {self.format!r}; choose from {FILE_FORMATS}"
            )

    def compile(self) -> re.Pattern:
        return re.compile(self.pattern)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "extensions": list(self.extensions),
            "format": self.format,
            "object_marker": self.object_marker,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FilePattern":
        return cls(
            pattern=data["pattern"],
            extensions=list(data.get("extensions", ["txt"])),
            format=data.get("format", "lines"),
            object_marker=data.get("object_marker", ""),
        )


@dataclass
class LineSchema:
    """
    Describes how one data line inside a matched file is parsed.

    Plain-column format (``key_sep=None``)::

        <v0><value_sep><v1><value_sep>...
        e.g. ``1;5000;3303``

    *metrics* maps column index (0-based) → record key name.  *cast* is a JSON
    name (``"int"``/``"float"``/``"str"``) applied to every column value.
    """

    key_sep: str | None = ": "  # None → plain value_sep columns
    value_sep: str = ";"
    line_key: str | None = None  # record key for the line prefix (keyed format only)
    metrics: dict[int, str] = field(default_factory=dict)
    cast: str = "int"

    def __post_init__(self) -> None:
        if self.key_sep is None and self.line_key is not None:
            raise ValueError(
                "line_key is ignored when key_sep is None (plain-column format)"
            )
        if self.cast not in CAST_TYPES:
            raise ValueError(f"Unknown cast {self.cast!r}; choose from {CAST_TYPES}")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "key_sep": self.key_sep,
            "value_sep": self.value_sep,
            # JSON object keys are strings; keep index order (insertion order).
            "metrics": {str(idx): name for idx, name in sorted(self.metrics.items())},
            "cast": self.cast,
        }
        if self.line_key is not None:
            out["line_key"] = self.line_key
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LineSchema":
        return cls(
            key_sep=data.get("key_sep", None),
            value_sep=data.get("value_sep", ";"),
            line_key=data.get("line_key"),
            metrics={int(idx): name for idx, name in data.get("metrics", {}).items()},
            cast=data.get("cast", "int"),
        )


@dataclass
class ScanSchema:
    """
    Full description of one data source: folder hierarchy, file naming, and
    line format.  Serializes to/from JSON (:meth:`~ScanSchema.to_dict`,
    :meth:`~ScanSchema.from_dict`) and is normally read via
    :func:`read_schemas`.
    """

    name: str = "default"
    folder_levels: list[FolderLevel] = field(default_factory=list)
    file_pattern: FilePattern = field(
        default_factory=lambda: FilePattern(pattern=r".*")
    )
    line_schema: LineSchema | None = None

    def __post_init__(self) -> None:
        if self.file_pattern.format == "lines" and self.line_schema is None:
            raise ValueError("line_schema is required when file_pattern.format='lines'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "folder_levels": [lvl.to_dict() for lvl in self.folder_levels],
            "file_pattern": self.file_pattern.to_dict(),
            "line_schema": self.line_schema.to_dict()
            if self.line_schema is not None
            else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScanSchema":
        return cls(
            name=str(data.get("name", "default")),
            folder_levels=[
                FolderLevel.from_dict(lvl) for lvl in data.get("folder_levels", [])
            ],
            file_pattern=FilePattern.from_dict(data["file_pattern"]),
            line_schema=LineSchema.from_dict(data["line_schema"])
            if data.get("line_schema") is not None
            else None,
        )


def read_schemas(source: str | Path | dict[str, Any]) -> dict[str, ScanSchema]:
    """
    Load one or more scan schemas from a JSON *source*.

    *source* is a JSON file path, a JSON string, or a parsed dict.  The object
    may be a single schema (→ ``{name: schema}``) or an object mapping source
    name → schema (→ returned as-is).  Returns ``{source_name: ScanSchema}``.
    """
    data: dict[str, Any]
    if isinstance(source, dict):
        data = source
    else:
        path = Path(source)
        text = path.read_text() if path.is_file() else str(source)
        data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError(
            "schema JSON must be an object: a single ScanSchema or {name: ScanSchema}"
        )

    # A single schema has a "file_pattern" key; a multi-schema object has
    # named children, each with one.
    if "file_pattern" in data:
        schema = ScanSchema.from_dict(data)
        return {schema.name: schema}
    return {name: ScanSchema.from_dict(child) for name, child in data.items()}


__all__ = [
    "CAST_TYPES",
    "CAST_TYPE_MAP",
    "FILE_FORMATS",
    "FilePattern",
    "FolderLevel",
    "LineSchema",
    "ScanSchema",
    "read_schemas",
]
