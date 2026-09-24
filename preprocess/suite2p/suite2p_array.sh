#!/bin/bash
# One array task owns one Suite2p session and one GPU.
# Usage is handled by: python -m preprocess.suite2p.pipeline.submit --help
#SBATCH --job-name=suite2p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=amperenodes
#SBATCH --time=12:00:00
#SBATCH --output=preprocess/suite2p/logs/suite2p_%A_%a.out
#SBATCH --error=preprocess/suite2p/logs/suite2p_%A_%a.err

SECONDS=0

set -euo pipefail

if [ "$#" -ne 6 ]; then
    echo "Expected: manifest dataframe raw_root output_root frame_rate conda_env" >&2
    exit 2
fi

manifest=$1
dataframe=$2
raw_root=$3
output_root=$4
frame_rate=$5
conda_env=$6

module load CUDA/12.2.0
module load cuDNN/8.9.2.26-CUDA-12.2.0
module load Anaconda3/2023.07-2
conda activate "${conda_env}"

cd "${SLURM_SUBMIT_DIR}"
python -m preprocess.suite2p.pipeline.run_session \
    --manifest "${manifest}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --dataframe "${dataframe}" \
    --raw-root "${raw_root}" \
    --output-root "${output_root}" \
    --frame-rate "${frame_rate}"

duration=$SECONDS
printf "Time elapsed: %02d:%02d:%02d\n" \
    $((duration/3600)) \
    $((duration%3600/60)) \
    $((duration%60))