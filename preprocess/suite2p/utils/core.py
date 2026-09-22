from __future__ import annotations

__author__ = ("Saul", "Nuria")

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import suite2p
import preprocess.syncronize_voltage_rec as svr

__all__ = [
    "get_settings",
    "obtain_bad_frames_from_voltage_rec",
    "prepare_ops_1st_pass",
    "process_1_session_suite2p_offline",
    "process_single_session",
]


def _require_suite2p() -> Any:
    if suite2p is None:
        raise RuntimeError(
            "Suite2p is not installed in the active Python environment. "
            "Activate the Suite2p environment before running the pipeline."
        )
    return suite2p


def get_settings(default_settings_dir: str | Path | None, settings: dict | None = None) -> dict:
    """Return a Suite2p settings dictionary, optionally loading a saved defaults file."""
    from utils.suite2p_v1_config import get_suite2p_holo_settings

    if settings is None:
        settings = _require_suite2p().default_settings()

    if default_settings_dir:
        default_path = Path(default_settings_dir)
        loaded = np.load(default_path / "default_settings.npy", allow_pickle=True)
        if isinstance(loaded, np.ndarray):
            if loaded.size == 0:
                return settings
            loaded = loaded.item(0) if hasattr(loaded, "item") else loaded[0]
        if isinstance(loaded, dict):
            return loaded

    return get_suite2p_holo_settings()


def obtain_bad_frames_from_voltage_rec(
    voltage_rec_paths: Sequence[str | Path],
    frame_rate: float,
    size_recordings: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return bad-frame indices and a boolean mask for a whole session."""

    indices: list[np.ndarray] = [np.array([], dtype=int)]
    len_recording = 0

    for voltage_recording, recording_size in zip(voltage_rec_paths, size_recordings):
        _, _, peaks_I1, _, _, _, _, peaks_I6, _, _ = (
            svr.obtain_peaks_voltage(str(voltage_recording), frame_rate, int(recording_size))
        )
        indices_for_6 = svr.obtain_indices_per_peaks(peaks_I1, peaks_I6)
        if len(indices_for_6) > 0 and indices_for_6[0] < 5:
            indices_for_6 = indices_for_6[1:]
        indices.append(indices_for_6 + len_recording)
        len_recording += int(recording_size)

    stim_index = (
        np.concatenate([idx for idx in indices if idx.size > 0])
        if any(idx.size > 0 for idx in indices)
        else np.array([], dtype=int)
    )
    if stim_index.size == 0:
        bad_frames_index = np.array([], dtype=int)
    else:
        bad_frames_index = np.unique(
            np.concatenate([stim_index - 1, stim_index, stim_index + 1])
        ).astype(int)

    bad_frames_bool = np.zeros(len_recording, dtype=bool)
    if bad_frames_index.size > 0:
        bad_frames_index.sort()
        bad_frames_bool[bad_frames_index] = True

    return bad_frames_index, bad_frames_bool


def prepare_ops_1st_pass(
    default_path: str | Path,
    ops_path: str | Path,
    bad_frames: np.ndarray = np.empty(0),
) -> dict:
    """Modify the default Suite2p ops file before the first pass."""
    default_dir = Path(default_path)
    aux_ops = np.load(default_dir / "default_ops.npy", allow_pickle=True)
    ops = aux_ops.take(0)
    if len(bad_frames) > 0:
        ops["badframes"] = bad_frames
    np.save(Path(ops_path), ops, allow_pickle=True)
    return ops


def process_1_session_suite2p_offline(
    default_path: str | Path,
    folder_suite2p: str | Path,
    folder_im_paths: Sequence[str | Path],
    voltage_rec_paths: Sequence[str | Path],
    size_recordings: Sequence[int],
    frame_rate: float,
):
    """Execute a single Suite2p session using the legacy v1 offline workflow."""
    if len(folder_im_paths) != len(size_recordings) or len(folder_im_paths) != len(voltage_rec_paths):
        raise ValueError("The sizes of the list need to be all the same")

    suite2p_runtime = _require_suite2p()
    folder_suite2p = Path(folder_suite2p)
    folder_suite2p.mkdir(parents=True, exist_ok=True)

    db = {
        "data_path": [str(path) for path in folder_im_paths],
        "save_path0": str(folder_suite2p),
        "fast_disk": str(folder_suite2p),
    }
    bad_frames, _ = obtain_bad_frames_from_voltage_rec(
        voltage_rec_paths,
        frame_rate,
        size_recordings,
    )
    np.save(Path(folder_im_paths[0]) / "bad_frames.npy", bad_frames)
    ops_1st_pass = prepare_ops_1st_pass(
        default_path,
        folder_suite2p / "ops_before_1st.npy",
        bad_frames,
    )
    ops_after_1st_pass = suite2p_runtime.run_s2p(ops_1st_pass, db)
    np.save(folder_suite2p / "ops_after_1st_pass.npy", ops_after_1st_pass, allow_pickle=True)

    return ops_after_1st_pass


def process_single_session(
    im_dirs: Sequence[str | Path],
    voltage_rec_dirs: Sequence[str | Path],
    size_recordings: Sequence[int],
    frame_rate: float,
    suite2p_save_path: str | Path,
    default_settings_dir: str | Path = "",
):
    """Process one session with the HoloBMI Suite2p configuration, merging the v1 and modern workflows."""
    if len(im_dirs) != len(size_recordings) or len(im_dirs) != len(voltage_rec_dirs):
        raise ValueError(
            "im_dirs, voltage_rec_dirs, and size_recordings must have the same length"
        )

    from utils.suite2p_v1_config import get_suite2p_holo_db

    suite2p_runtime = _require_suite2p()
    bad_frames, bad_frames_bool = obtain_bad_frames_from_voltage_rec(
        voltage_rec_dirs,
        frame_rate,
        size_recordings,
    )
    db = get_suite2p_holo_db(
        [str(path) for path in im_dirs],
        suite2p_save_path,
        bad_frames,
        bad_frames_bool,
    )
    settings = get_settings(default_settings_dir)
    return suite2p_runtime.run_s2p(db, settings)
