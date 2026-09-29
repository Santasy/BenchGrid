"""
Polars-native scanning for BenchGrid.

Both fast paths are all-vectorized: the whole tree is read in one call and the
per-file metadata (folder levels + file-pattern named groups) is extracted from
the scanned path, so there is no per-file Python loop.

- :func:`scan_csv_lazy` — plain-column ``format="lines"`` sources (one glob per
  extension, ``pl.scan_csv``), returns a lazy frame; chain filters/aggregations
  and ``.collect()`` once.
- :func:`scan_json` — ``format="json"`` object streams (the "bracketless
  array" formulation: objects joined by commas, no ``[...]``), collapsed to
  NDJSON and parsed with a single ``pl.read_ndjson``.  Nested values stay
  nested (sub-objects become struct columns, arrays become list columns).
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

import polars as pl

from .schema import CAST_TYPE_MAP, ScanSchema


def scan_csv_lazy(
    root_dirs: list[str],
    schema: ScanSchema,
    *,
    path_column: str = "__path__",
):
    """
    Scan plain-column CSV trees with Polars and return a lazy frame.

    *schema* must be ``format == "lines"`` with ``LineSchema.key_sep is None``
    (plain-column data); folder levels and file pattern must use named groups,
    which become the row metadata columns.  Folder/tag regex groups come out
    as strings (matching the legacy ``scan`` semantics); line values are typed
    with ``LineSchema.cast``.
    """
    fp = schema.file_pattern
    ls = schema.line_schema
    if fp.format != "lines":
        raise ValueError(
            f"scan_csv_lazy supports format='lines' only (got {fp.format!r}). "
            "Use scan_json() for JSON object streams."
        )
    if ls is None or ls.key_sep is not None:
        raise ValueError(
            "scan_csv_lazy requires plain-column lines (LineSchema.key_sep=None). "
            "Use scan() for keyed-line sources."
        )

    metric_cols = [ls.metrics[i] for i in sorted(ls.metrics)]
    if not metric_cols:
        raise ValueError(
            f"scan_csv_lazy needs LineSchema.metrics columns, got none: {ls.metrics!r}"
        )

    path_groups = [_strip_anchors(lvl.pattern) for lvl in schema.folder_levels] + [
        _strip_anchors(fp.pattern)
    ]
    path_group_names = {
        name for group in path_groups for name in re.findall(r"\?P<(\w+)>", group)
    }
    if path_group_names & set(metric_cols):
        raise ValueError(
            "scan_csv_lazy: named groups in folder/file patterns must not collide "
            "with metric columns. "
            f"conflicts={sorted(path_group_names & set(metric_cols))!r}"
        )

    globs = [
        f"{str(root).rstrip('/')}/**/*.{ext.lstrip('.')}"
        for root in root_dirs
        for ext in (fp.extensions or ["txt"])
    ]
    overrides = {
        col: dtype
        for col in metric_cols
        for dtype in (_cast_dtype(ls.cast),)
        if dtype is not None
    }

    lf = pl.scan_csv(
        globs,
        has_header=False,
        new_columns=metric_cols,
        separator=ls.value_sep or ";",
        include_file_paths=path_column,
        schema_overrides=overrides,
        ignore_errors=True,
    )
    return (
        lf.with_columns(
            pl.col(path_column)
            .str.extract_groups(_csv_path_regex(root_dirs, schema))
            .alias("__meta__")
        )
        .unnest("__meta__")
        .drop(path_column)
    )


def scan_csv(root_dirs: list[str], schema: ScanSchema, **kwargs):
    """Eager variant of :func:`scan_csv_lazy`. Collects to a DataFrame."""
    return scan_csv_lazy(root_dirs, schema, **kwargs).collect()


def scan_json(root_dirs: list[str], schema: ScanSchema) -> pl.DataFrame:
    """
    Polars fast path for JSON object streams (``FilePattern(format="json")``).

    The files are a "bracketless array": objects joined by commas (or just
    newlines) with no surrounding ``[...]`` and often a trailing comma, e.g.
    ``{"step":1,...},{"step":2,...},``.  Each file is collapsed to one line per
    object, the per-file metadata keys (folder levels + file-pattern named
    groups) are embedded into every object, and the whole tree is parsed in a
    single ``pl.read_ndjson`` call.  Nested values stay nested: sub-objects
    become struct columns and arrays become list columns.
    """
    fp = schema.file_pattern
    if fp.format != "json":
        raise ValueError(
            f"scan_json supports format='json' only (got {fp.format!r}). "
            "Use scan_csv_lazy() for line-based sources."
        )
    marker = fp.object_marker
    if not marker:
        raise ValueError(
            "scan_json requires FilePattern.object_marker — the name of the "
            "key that opens every JSON object (e.g. 'step')."
        )

    marker_token = '{"' + marker + '"'
    compiled_folders = [lvl.compile() for lvl in schema.folder_levels]
    compiled_file = fp.compile()

    parts: list[str] = []
    for root in root_dirs:
        root_path = Path(root)
        if not root_path.is_dir():
            print(f"[scan] skipping non-existent root: {root}")
            continue
        for path in sorted(root_path.rglob("*")):
            if not path.is_file():
                continue
            if fp.extensions and path.suffix.lstrip(".") not in fp.extensions:
                continue
            keys = _file_keys_from_path(
                path, root_path, schema, compiled_folders, compiled_file
            )
            if keys is None:
                continue  # does not match the folder/file layout
            parts.append(_json_stream_to_ndjson(path.read_text(), marker_token, keys))

    buffer = "\n".join(p for p in parts if p)
    if not buffer:
        return pl.DataFrame()
    return pl.read_ndjson(io.StringIO(buffer))


def _file_keys_from_path(
    path: Path,
    root_path: Path,
    schema: ScanSchema,
    compiled_folders: list[re.Pattern],
    compiled_file: re.Pattern,
) -> dict[str, Any] | None:
    """Folder-level + file-pattern named groups for one file, or None if the
    file does not match the schema layout."""
    rel = path.relative_to(root_path).parts
    if len(rel) < len(schema.folder_levels) + 1:
        return None
    keys: dict[str, Any] = {}
    for folder_level, pattern, part in zip(
        schema.folder_levels, compiled_folders, rel[: len(schema.folder_levels)]
    ):
        m = pattern.fullmatch(part)
        if m is None:
            if folder_level.required:
                return None
            continue  # non-required level: pass through, no keys
        keys.update(m.groupdict())
    m = compiled_file.fullmatch(rel[-1])
    if m is None:
        return None
    keys.update(m.groupdict())
    return keys


def _json_stream_to_ndjson(raw: str, marker_token: str, keys: dict[str, Any]) -> str:
    """Collapse one "bracketless array" file into NDJSON (one object per line).

    Whitespace is dropped (objects become single lines), a trailing comma is
    stripped before the final object, and the schema's per-file metadata keys
    are injected at the start of every object.  The top-level separator is
    detected as ``},<marker_token>`` (the marker never appears inside nested
    values).
    """
    flat = "".join(raw.split()).rstrip(",")
    if not flat:
        return ""
    meta_body = json.dumps(keys, separators=(",", ":"))[1:-1]
    obj_open = (
        marker_token if not meta_body else "{" + meta_body + "," + marker_token[1:]
    )
    tagged = flat.replace(marker_token, obj_open)
    # objects separated by a comma (usual) or just adjacent braces (tolerated)
    for boundary in ("}," + obj_open, "}" + obj_open):
        tagged = tagged.replace(boundary, "}\n" + obj_open)
    return tagged


def _strip_anchors(pattern: str) -> str:
    """Drop a leading ``^`` / trailing ``$`` so a level pattern embeds in a bigger regex."""
    if pattern.startswith("^"):
        pattern = pattern[1:]
    if pattern.endswith("$") and not pattern.endswith("\\$"):
        pattern = pattern[:-1]
    return pattern


def _csv_path_regex(root_dirs: list[str], schema: ScanSchema) -> str:
    """One anchored regex for the whole tree: ``^<root(s)>/<level0>/.../<file>$``."""
    body = "/".join(
        [_strip_anchors(lvl.pattern) for lvl in schema.folder_levels]
        + [_strip_anchors(schema.file_pattern.pattern)]
    )
    prefix = (
        re.escape(str(root_dirs[0]).rstrip("/"))
        if len(root_dirs) == 1
        else "(?:%s)" % "|".join(re.escape(str(root).rstrip("/")) for root in root_dirs)
    )
    return f"^{prefix}/{body}$"


def _cast_dtype(cast: str):
    t = CAST_TYPE_MAP.get(cast)
    if t is int:
        return pl.Int64
    if t is float:
        return pl.Float64
    if t is str:
        return pl.Utf8
    return None


__all__ = ["scan_csv", "scan_csv_lazy", "scan_json"]
