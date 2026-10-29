"""Scan/plot smoke tests for the string-heavy paths, on a throwaway tree.

Guards the crash-prone readers (the ones that shipped broken because nothing
ran them): the shared tree walker, the JSON stream parser and the text-less
trade-off derive. Runs headless, no streamlit, no framework.
"""

import tempfile
from pathlib import Path

from benchgrid.data import Dataset
from benchgrid.plots.adapter import PlotsAdapter
from benchgrid.plots.context import TabContext
from benchgrid.plots.pairs import PairPlotSpec, pair_frame
from benchgrid.scan import scan_csv_lazy, scan_json
from benchgrid.schema import ScanSchema

csv_schema = ScanSchema.from_dict(
    {
        "name": "csv",
        "folder_levels": [{"pattern": r"^r(?P<refinement>\d+)$"}],
        "file_pattern": {"pattern": r"^(?P<variant>\w+)\.log$", "extensions": [".log"]},
        "line_schema": {
            "key_sep": None,
            "value_sep": ";",
            "metrics": {"0": "duration_ms", "1": "bytes"},
            "cast": "int",
        },
    }
)
json_schema = ScanSchema.from_dict(
    {
        "name": "json",
        "folder_levels": [{"pattern": r"^r(?P<refinement>\d+)$"}],
        "file_pattern": {
            "pattern": r"^(?P<variant>\w+)\.json$",
            "extensions": ["json"],
            "format": "json",
            "object_marker": "step",
        },
    }
)

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / "r5").mkdir()
    (root / "r5" / "A.log").write_text("100;2000\n150;2100\n")
    (root / "r5" / "B.log").write_text("90;1900\n")
    (root / "r5" / "nohup.out").write_text("noise\n")  # wrong extension, skipped
    # Spaced string value: the JSON reader must not eat the spaces.
    (root / "r5" / "A.json").write_text(
        '{"step": "gen all", "n": 1}, {"step": "ref near", "n": 2},'
    )

    csv = scan_csv_lazy([str(root)], csv_schema).collect()
    assert csv.height == 3, csv.height
    assert set(csv["variant"].to_list()) == {"A", "B"}
    assert csv["duration_ms"].to_list() == [100, 150, 90]
    assert csv["refinement"].to_list() == ["5", "5", "5"]

    js = scan_json([str(root)], json_schema)
    assert js.height == 2, js.height
    assert js["step"].to_list() == ["gen all", "ref near"], js["step"].to_list()
    assert js["n"].to_list() == [1, 2]

    # Text-less trade-off (PairPlotSpec.text_key=None) derives a frame and, as
    # in render_pairs, no row is looked up by a None column.
    metric_source = {"duration_ms": "csv", "bytes": "csv"}
    adapter = PlotsAdapter(
        title="t",
        schema_file="",
        data_dirs=[str(root)],
        metrics=["duration_ms", "bytes"],
        metric_source=metric_source,
        source_metrics={"csv": ["duration_ms", "bytes"]},
        run_keys=["variant"],
        x_keys=["variant"],
        y_key="bytes",
        progress_key=None,
        step_key=None,
    )
    ds = Dataset(
        frames={"csv": csv},
        schemas={"csv": csv_schema},
        metric_keys={"csv": frozenset(["duration_ms", "bytes"])},
    )
    ctx = TabContext(adapter=adapter, dataset=ds)
    spec = PairPlotSpec(
        x_metric="duration_ms",
        y_metric="bytes",
        point_keys=["variant"],
        color_keys=["variant"],
        text_key=None,
    )
    frame = pair_frame(ctx, spec)
    assert frame.height == 2, frame.height

print("ok")
