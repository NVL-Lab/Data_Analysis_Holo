"""Run one NWB conversion session selected by a manifest entry."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from preprocess.nwb.pipeline.manifest import read_manifest_entry


def run_session(
    manifest_path: Path,
    task_id: int,
    dataframe_path: Path,
    raw_root: Path,
    output_root: Path,
    behavior_root: Path | None = None,
) -> None:
    """Convert a single dataframe row into NWB files."""
    entry = read_manifest_entry(manifest_path, task_id)
    dataframe = pd.read_parquet(dataframe_path).reset_index(drop=True)
    row_index = int(entry["row_index"])
    if row_index >= len(dataframe):
        raise IndexError(f"Manifest row index {row_index} is not in {dataframe_path}")

    row = dataframe.iloc[row_index]
    if str(row["session_path"]) != str(entry["session_path"]):
        raise ValueError(
            "The dataframe changed after the manifest was created; "
            f"expected session {entry['session_path']!r}, "
            f"found {row['session_path']!r}"
        )

    try:
        from nwb.convert_to_nwb import convert_experiment_to_nwb
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "NWB conversion dependencies are unavailable. Activate the correct "
            "NWB environment before running a session."
        ) from exc

    behavior_dir = behavior_root or raw_root.parent / "Behavior"
    convert_experiment_to_nwb(
        row_index,
        str(output_root),
        str(raw_root),
        str(behavior_dir),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--dataframe", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--behavior-root",
        type=Path,
        default=None,
        help="Optional behavior data root; defaults to Raw/../Behavior",
    )
    args = parser.parse_args()
    run_session(
        args.manifest,
        args.task_id,
        args.dataframe,
        args.raw_root,
        args.output_root,
        args.behavior_root,
    )


if __name__ == "__main__":
    main()
