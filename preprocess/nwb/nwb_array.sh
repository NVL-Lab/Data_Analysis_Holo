#!/bin/bash
# One array task owns one NWB session conversion.
# Usage is handled by: python -m preprocess.nwb.pipeline.submit --help
#SBATCH --job-name=nwb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=amperenodes
#SBATCH --time=12:00:00
#SBATCH --output=preprocess/nwb/logs/nwb_%A_%a.out
#SBATCH --error=preprocess/nwb/logs/nwb_%A_%a.err

set -euo pipefail

if [ "$#" -ne 6 ]; then
    echo "Expected: manifest dataframe raw_root output_root behavior_root conda_env" >&2
    exit 2
fi

manifest=$1
dataframe=$2
raw_root=$3
output_root=$4
behavior_root=$5
conda_env=$6

module load Anaconda3/2023.07-2
conda activate "${conda_env}"

cd "${SLURM_SUBMIT_DIR}"
python -m preprocess.nwb.pipeline.run_session \
    --manifest "${manifest}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --dataframe "${dataframe}" \
    --raw-root "${raw_root}" \
    --output-root "${output_root}" \
    --behavior-root "${behavior_root}"
