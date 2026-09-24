"""Create a Suite2p manifest and submit it as a SLURM array."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

from preprocess.suite2p.pipeline.manifest import write_manifest


def _print_parameters(parameters: argparse.Namespace) -> None:
    print("Suite2p pipeline parameters:")
    for name, value in sorted(vars(parameters).items()):
        print(f"  {name}={value!r}")


def _day_index(value: str) -> str:
    if not re.fullmatch(r"D.*", value):
        raise argparse.ArgumentTypeError(
            "session day must match the pattern 'D*' (start with 'D')"
        )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataframe", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-rate", type=float, required=True)
    parser.add_argument(
        "--executor",
        choices=("slurm", "local"),
        default="slurm",
        help="Run through SLURM or process sessions in the current environment",
    )
    parser.add_argument(
        "--conda-env",
        default=os.environ.get("SUITE2P_CONDA_ENV"),
        help="Conda environment for SLURM tasks (or SUITE2P_CONDA_ENV)",
    )
    parser.add_argument("--session-date")
    parser.add_argument("--day-index", type=_day_index, default=None)
    parser.add_argument("--mouse-id")
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Maximum concurrent SLURM tasks; defaults to the number of selected sessions",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("preprocess/suite2p/manifests"),
    )
    parser.add_argument(
        "--slurm-script",
        type=Path,
        default=Path("preprocess/suite2p/suite2p_array.sh"),
    )
    args = parser.parse_args()
    _print_parameters(args)

    if args.max_parallel is not None and args.max_parallel < 1:
        parser.error("--max-parallel must be at least 1")
    if args.executor == "slurm" and not args.conda_env:
        parser.error("--conda-env is required unless SUITE2P_CONDA_ENV is set")
    if not args.dataframe.is_file():
        parser.error(f"Dataframe does not exist: {args.dataframe}")
    if not args.raw_root.is_dir():
        parser.error(f"Raw data directory does not exist: {args.raw_root}")
    if not args.slurm_script.is_file():
        parser.error(f"SLURM script does not exist: {args.slurm_script}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = args.manifest_dir / f"suite2p_{timestamp}.jsonl"
    session_count = write_manifest(
        args.dataframe,
        manifest_path,
        args.session_date,
        args.day_index,
        args.mouse_id,
    )
    max_parallel = session_count if args.max_parallel is None else min(args.max_parallel, session_count)

    if args.executor == "local":
        from preprocess.suite2p.pipeline.run_session import run_session

        for task_id in range(session_count):
            run_session(
                manifest_path,
                task_id,
                args.dataframe,
                args.raw_root,
                args.output_root,
                args.frame_rate,
            )
        print(f"Processed sessions: {session_count}")
        print(f"Manifest: {manifest_path}")
        return

    Path("preprocess/suite2p/logs").mkdir(parents=True, exist_ok=True)

    array = f"0-{session_count - 1}%{max_parallel}"
    command = [
        "sbatch",
        f"--array={array}",
        str(args.slurm_script),
        str(manifest_path),
        str(args.dataframe),
        str(args.raw_root),
        str(args.output_root),
        str(args.frame_rate),
        args.conda_env,
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print(f"Manifest: {manifest_path}")
    print(f"Selected sessions: {session_count}")
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
