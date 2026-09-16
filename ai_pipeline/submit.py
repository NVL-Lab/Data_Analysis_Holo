"""Create a Suite2p manifest and submit it as a SLURM array."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

from ai_pipeline.manifest import write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataframe", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-rate", type=float, required=True)
    parser.add_argument("--session-date")
    parser.add_argument("--mouse-id")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument(
        "--manifest-dir", type=Path, default=Path("manifests")
    )
    parser.add_argument(
        "--slurm-script",
        type=Path,
        default=Path("ai_pipeline/slurm/suite2p_array.sh"),
    )
    args = parser.parse_args()

    if args.max_parallel < 1:
        parser.error("--max-parallel must be at least 1")
    if not args.dataframe.is_file():
        parser.error(f"Dataframe does not exist: {args.dataframe}")
    if not args.raw_root.is_dir():
        parser.error(f"Raw data directory does not exist: {args.raw_root}")
    if not args.slurm_script.is_file():
        parser.error(f"SLURM script does not exist: {args.slurm_script}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = args.manifest_dir / f"suite2p_{timestamp}.jsonl"
    session_count = write_manifest(
        args.dataframe, manifest_path, args.session_date, args.mouse_id
    )
    Path("logs").mkdir(exist_ok=True)

    array = f"0-{session_count - 1}%{args.max_parallel}"
    command = [
        "sbatch",
        f"--array={array}",
        str(args.slurm_script),
        str(manifest_path),
        str(args.dataframe),
        str(args.raw_root),
        str(args.output_root),
        str(args.frame_rate),
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print(f"Manifest: {manifest_path}")
    print(f"Selected sessions: {session_count}")
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
