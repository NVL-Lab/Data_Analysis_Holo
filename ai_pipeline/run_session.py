"""Run one Suite2p session selected by a manifest entry."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ai_pipeline.manifest import read_manifest_entry
from preprocess.preprocess_suite2p_v1 import process_single_session


IMAGE_COLUMNS = (
    "holostim_seq_im_path",
    "baseline_im_path",
    "pretrain_im_path",
    "bmi_im_path",
)
VOLTAGE_COLUMNS = (
    "holostim_seq_voltage_file",
    "baseline_voltage_file",
    "pretrain_voltage_file",
    "bmi_voltage_file",
)


def _required_paths(
    row: pd.Series, raw_root: Path
) -> tuple[list[str], list[str], list[int]]:
    session_root = raw_root / str(row["session_path"])
    image_paths = [session_root / str(row[column]) for column in IMAGE_COLUMNS]
    voltage_paths = [session_root / str(row[column]) for column in VOLTAGE_COLUMNS]

    missing = [path for path in [*image_paths, *voltage_paths] if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Input files are missing:\n{formatted}")

    recording_sizes = [
        len(tuple(path.glob("*.tif"))) for path in image_paths
    ]
    if any(size == 0 for size in recording_sizes):
        raise ValueError(f"No TIFF frames found in {session_root}")

    return (
        [str(path) for path in image_paths],
        [str(path) for path in voltage_paths],
        recording_sizes,
    )


def run_session(
    manifest_path: Path,
    task_id: int,
    dataframe_path: Path,
    raw_root: Path,
    output_root: Path,
    frame_rate: float,
) -> None:
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

    image_paths, voltage_paths, recording_sizes = _required_paths(row, raw_root)
    output_path = output_root / str(row["session_path"])
    output_path.mkdir(parents=True, exist_ok=True)

    process_single_session(
        image_paths,
        voltage_paths,
        recording_sizes,
        frame_rate,
        output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--dataframe", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-rate", type=float, required=True)
    args = parser.parse_args()
    run_session(
        args.manifest,
        args.task_id,
        args.dataframe,
        args.raw_root,
        args.output_root,
        args.frame_rate,
    )


if __name__ == "__main__":
    main()
