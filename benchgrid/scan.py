"""
Polars-native scanning for BenchGrid.

Two vectorized fast paths read the whole tree in one call and extract the
per-file metadata (folder levels + file-pattern named groups) from the scanned
path, so there is no per-file Python loop:

- :func:`scan_csv_lazy` — plain-column ``format="lines"`` sources (the tree is
  resolved to the schema's matching files first, then ``pl.scan_csv``), returns
  a lazy frame; chain filters/aggregations and ``.collect()`` once.
  ``LineSchema.ragged`` switches the read to whole-line + positional split, so
  a tree whose rows have different widths still scans in one pass.
- :func:`scan_json` — ``format="json"`` object streams (the "bracketless
  array" formulation: objects joined by commas, no ``[...]``), collapsed to
  NDJSON and parsed with a single ``pl.read_ndjson``.  Nested values stay
  nested (sub-objects become struct columns, arrays become list columns).

A third path, :func:`scan_csv_with_key_separator`, covers key-separated lines
(``LineSchema.key_sep`` set): each line is ``<key><sep><values...>`` and the
keyed prefix is not a plain column ``pl.scan_csv`` can read, so this path
walks the tree in Python — reusing the same folder/file matching as the fast
paths — parses one line at a time, and assembles the rows into an eager
``pl.DataFrame``.  It is the only non-vectorized path and the only one that
returns an eager frame by construction (there is no lazy variant).
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

import polars as pl

from .schema import CAST_TYPE_MAP, ScanSchema

#: Separator that cannot occur in a data line, so a CSV row is read as one
#: field and split positionally (the ragged read path).
_LINE_SENTINEL = "\x00"


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

    Only files that match the schema layout are read: sibling logs of another
    shape (``nohup``, ``itall``, …) are filtered out *before* the read, so they
    cannot break the column schema.  A tree with no matching file yields an
    empty frame carrying the schema's columns.

    With ``LineSchema.ragged`` the rows of one tree may have different widths:
    the tree is read whole-line and split on ``value_sep``, then column *i* is
    ``null`` where a row is shorter.  Unknown trailing fields are dropped.  A
    fixed-width ``pl.scan_csv`` cannot read such a tree at all (it infers one
    width from the first file and fails on the others), so this path reads every
    file identically instead of branching per file.
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

    path_groups = [
        _strip_anchors(lvl.pattern) for lvl in schema.folder_levels
    ] + [_strip_anchors(fp.pattern)]
    path_group_names = {
        name
        for group in path_groups
        for name in re.findall(r"\?P<(\w+)>", group)
    }
    if path_group_names & set(metric_cols):
        raise ValueError(
            "scan_csv_lazy: named groups in folder/file patterns must not collide "
            "with metric columns. "
            f"conflicts={sorted(path_group_names & set(metric_cols))!r}"
        )

    paths = [
        str(path)
        for path, _ in _iter_matching_files(root_dirs, schema)
        if ls.ragged or _line_width_matches(path, ls, len(metric_cols))
    ]
    if not paths:
        return _empty_lazy_frame(schema, metric_cols)

    if ls.ragged:
        lf = _scan_ragged(paths, metric_cols, ls, path_column)
    else:
        dtype = _cast_dtype(ls.cast)
        lf = pl.scan_csv(
            paths,
            has_header=False,
            new_columns=metric_cols,
            separator=ls.value_sep or ";",
            include_file_paths=path_column,
            schema_overrides={col: dtype for col in metric_cols}
            if dtype
            else None,
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


def _iter_matching_files(root_dirs: list[str], schema: ScanSchema):
    """
    Yield ``(path, keys)`` for the files under *root_dirs* matching *schema*.

    The one walker all three readers share: folder-level + file-pattern named
    groups become *keys* (folder levels without named groups, and optional
    levels whose pattern misses, contribute nothing).  Extension normalisation
    is one rule — a leading dot in ``FilePattern.extensions`` is ignored — so
    the fast and eager paths can never disagree.  A scan reads only matching
    files, so a sibling log of another shape never reaches the reader.
    """
    compiled_folders = [lvl.compile() for lvl in schema.folder_levels]
    compiled_file = schema.file_pattern.compile()
    extensions = {
        ext.lstrip(".") for ext in (schema.file_pattern.extensions or ["txt"])
    }
    depth = len(schema.folder_levels) + 1
    for root in root_dirs:
        root_path = Path(root)
        if not root_path.is_dir():
            print(f"[scan] skipping non-existent root: {root}")
            continue
        for path in sorted(root_path.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lstrip(".") not in extensions:
                continue
            rel = path.relative_to(root_path).parts
            if len(rel) != depth:
                continue
            keys = _file_keys_from_parts(
                rel, schema, compiled_folders, compiled_file
            )
            if keys is not None:
                yield path, keys


def _file_keys_from_parts(
    rel: tuple[str, ...],
    schema: ScanSchema,
    compiled_folders: list[re.Pattern],
    compiled_file: re.Pattern,
) -> dict[str, Any] | None:
    """Folder-level + file-pattern named groups for one *rel* path, or None
    when the path does not match the schema layout."""
    keys: dict[str, Any] = {}
    for folder_level, pattern, part in zip(
        schema.folder_levels, compiled_folders, rel[: len(schema.folder_levels)]
    ):
        m = pattern.fullmatch(part)
        if m is None:
            if folder_level.required:
                return None
            continue  # optional level: pass through, no keys
        keys.update(m.groupdict())
    m = compiled_file.fullmatch(rel[-1])
    if m is None:
        return None
    keys.update(m.groupdict())
    return keys


def _line_width_matches(path: Path, line_schema: Any, width: int) -> bool:
    """Whether the first non-empty line has *width* separator-split fields."""
    sep = (line_schema.value_sep if line_schema is not None else None) or ";"
    try:
        with path.open() as fh:
            for raw in fh:
                line = raw.strip()
                if line:
                    return len(line.split(sep)) == width
    except OSError as exc:
        print(f"[scan] skipping unreadable file {path}: {exc}")
        return False
    return True


def _empty_lazy_frame(
    schema: ScanSchema, metric_cols: list[str]
) -> pl.LazyFrame:
    """An empty frame carrying the scan's full column set (no matching file)."""
    ls = schema.line_schema
    metric_dtype = _cast_dtype(ls.cast) if ls is not None else None
    fields: dict[str, Any] = {}
    for pattern in [
        *(lvl.pattern for lvl in schema.folder_levels),
        schema.file_pattern.pattern,
    ]:
        for name in re.findall(r"\?P<(\w+)>", pattern):
            fields[name] = pl.Utf8
    for name in metric_cols:
        fields[name] = metric_dtype or pl.Utf8
    return pl.LazyFrame(schema=fields)


def _scan_ragged(
    paths: list[str],
    metric_cols: list[str],
    line_schema: Any,
    path_column: str,
) -> Any:
    """
    Whole-line read + positional split for a tree of mixed-width rows.

    Each line is read as a single field (a separator that cannot occur in the
    data), split on ``value_sep`` once, and column *i* is taken positionally
    with ``null`` when the row is shorter — still one pass, no per-file loop.
    """
    lf = pl.scan_csv(
        paths,
        has_header=False,
        new_columns=[_LINE_KEY],
        separator=_LINE_SENTINEL,
        include_file_paths=path_column,
        ignore_errors=True,
    ).with_columns(
        pl.col(_LINE_KEY)
        .str.split(line_schema.value_sep or ";")
        .alias(_FIELDS_KEY)
    )
    return lf.with_columns(
        *[
            pl.col(_FIELDS_KEY).list.get(index, null_on_oob=True).alias(name)
            for index, name in enumerate(metric_cols)
        ]
    ).drop(_FIELDS_KEY, _LINE_KEY)


#: Internal column names of the ragged read path.
_LINE_KEY = "__line__"
_FIELDS_KEY = "__fields__"


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

    parts: list[str] = []
    for path, keys in _iter_matching_files(root_dirs, schema):
        parts.append(
            _json_stream_to_ndjson(path.read_text(), marker_token, keys)
        )

    buffer = "\n".join(p for p in parts if p)
    if not buffer:
        return pl.DataFrame()
    return pl.read_ndjson(io.StringIO(buffer))


def scan_csv_with_key_separator(
    root_dirs: list[str], schema: ScanSchema
) -> pl.DataFrame:
    """
    Walk each directory in *root_dirs* according to *schema* and return a
    flat polars DataFrame.  Each row is built by the same per-line reading
    logic, with columns fully determined by the schema.
    """
    fp = schema.file_pattern
    ls = schema.line_schema
    if fp.format != "lines":
        raise ValueError(
            f"scan_csv_with_key_separator supports format='lines' only "
            f"(got {fp.format!r}). Use scan_json() for JSON object streams."
        )
    if ls is None or ls.key_sep is None:
        raise ValueError(
            "scan_csv_with_key_separator requires a key-separator"
            " per line (LineSchema.key_sep). "
        )

    records: list[dict] = []
    for path, file_keys in _iter_matching_files(root_dirs, schema):
        _parse_keyed_lines(path, schema.line_schema, file_keys, records)

    return pl.DataFrame(records)


def _parse_keyed_lines(
    filepath: Path,
    ls: Any,
    file_keys: dict[str, Any],
    records: list[dict],
) -> None:
    """Parse one ``<key><key_sep><values...>`` file into *records*."""
    csv_metrics = dict(ls.metrics)
    csv_metrics.pop(0)  # Pop the line_key at index 0.
    cast = CAST_TYPE_MAP.get(ls.cast)

    with open(filepath) as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split(ls.key_sep, 1)
            if len(parts) != 2:
                continue
            try:
                record: dict[str, Any] = {
                    **file_keys,
                    ls.line_key: cast(parts[0]),
                }
                values = parts[1].split(ls.value_sep)
                for idx, name in csv_metrics.items():
                    record[name] = cast(values[idx - 1])
                records.append(record)
            except (ValueError, IndexError) as exc:
                print(f"[scan] {filepath}:{lineno}: {exc}")


def _json_stream_to_ndjson(
    raw: str, marker_token: str, keys: dict[str, Any]
) -> str:
    """Collapse one "bracketless array" file into NDJSON (one object per line).

    Whitespace *between* objects is dropped but whitespace inside string
    values is preserved (``pl.read_ndjson`` skips it between tokens).  A
    trailing comma is stripped before the final object, and the schema's
    per-file metadata keys are injected at the start of every object.  The
    top-level separator is detected as ``},<marker_token>`` (the marker never
    appears inside nested values).
    """
    flat = raw.rstrip().rstrip(",")
    if not flat.strip():
        return ""
    # Collapse whitespace between tokens only, leaving quoted strings intact:
    # splitting on quotes keeps every odd chunk (a string body) verbatim.
    chunks = flat.split('"')
    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            chunks[i] = "".join(chunk.split())
    flat = '"'.join(chunks)
    meta_body = json.dumps(keys, separators=(",", ":"))[1:-1]
    obj_open = (
        marker_token
        if not meta_body
        else "{" + meta_body + "," + marker_token[1:]
    )
    tagged = flat.replace(marker_token, obj_open)
    # objects separated by a comma (usual) or just adjacent braces (tolerated)
    for boundary in ("}," + obj_open, "}" + obj_open):
        tagged = tagged.replace(boundary, "}\n" + obj_open)
    return tagged


def _strip_anchors(pattern: str) -> str:
    """Drop a leading ``^`` / trailing ``$`` so a level pattern embeds in a bigger regex."""
    pattern = pattern.removeprefix("^")
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
        else "(?:{})".format("|".join(re.escape(str(root).rstrip("/")) for root in root_dirs))
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


__all__ = [
    "scan_csv",
    "scan_csv_lazy",
    "scan_csv_with_key_separator",
    "scan_json",
]
