"""Create and read immutable Suite2p session manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {
    "session_path",
    "holostim_seq_im_path",
    "baseline_im_path",
    "pretrain_im_path",
    "bmi_im_path",
    "holostim_seq_voltage_file",
    "baseline_voltage_file",
    "pretrain_voltage_file",
    "bmi_voltage_file",
}


def select_rows(
    dataframe_path: Path,
    session_date: str | None = None,
    day_index: str | None = None,
    mouse_id: str | None = None,
) -> pd.DataFrame:
    """Load the dataframe and select sessions using explicit filters."""
    dataframe = pd.read_parquet(dataframe_path).reset_index(drop=True)
    missing = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"{dataframe_path} is missing required columns: {', '.join(sorted(missing))}"
        )

    selected = dataframe
    if session_date is not None:
        selected = selected[selected["session_date"].eq(session_date)]
    if day_index is not None:
        selected = selected[selected["day_index"].eq(day_index)]
    if mouse_id is not None:
        selected = selected[selected["mouse_id"].eq(mouse_id)]

    return selected


def write_manifest(
    dataframe_path: Path,
    manifest_path: Path,
    session_date: str | None = None,
    day_index: str | None = None,
    mouse_id: str | None = None,
) -> int:
    """Write one JSON object per selected dataframe row."""
    selected = select_rows(dataframe_path, session_date, day_index, mouse_id)
    if selected.empty:
        raise ValueError("The filters selected no sessions")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for row_index, row in selected.iterrows():
            manifest.write(
                json.dumps(
                    {"row_index": int(row_index), "session_path": row["session_path"]}
                )
                + "\n"
            )
    return len(selected)


def read_manifest_entry(manifest_path: Path, task_id: int) -> dict:
    """Return the zero-based manifest entry for a SLURM array task."""
    entries = manifest_path.read_text(encoding="utf-8").splitlines()
    if task_id < 0 or task_id >= len(entries):
        raise IndexError(
            f"Task ID {task_id} is outside manifest range 0-{max(len(entries) - 1, 0)}"
        )
    entry = json.loads(entries[task_id])
    if not isinstance(entry, dict) or "row_index" not in entry:
        raise ValueError(f"Invalid manifest entry at task ID {task_id}")
    return entry
