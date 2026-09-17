# Suite2p array pipeline

This pipeline creates a manifest of selected sessions and submits one SLURM
array task per session. Each task receives its own GPU allocation and runs one
Suite2p process; sessions do not compete for a GPU inside one job.

Run from the repository root on the cluster:

```bash
module load Anaconda3/2023.07-2
conda activate suite2p_v1

python -m ai_pipeline.submit \
  --dataframe /path/to/holobmi_df.parquet \
  --raw-root /data/project/nvl_lab/HoloBMI/Raw \
  --output-root /data/project/nvl_lab/processed_suite2p \
  --frame-rate 29.752 \
  --session-date 190930 \
  --day_index D10 \
  --mouse-id NVI12 \
  --max-parallel 2
```

`--session-date`, `--day_index`, and `--mouse-id` are optional filters.
`--day_index`, when provided, must start with `D` (for example, `D10`).
`--max-parallel` limits the number of simultaneously running array tasks.
The command prints the manifest path and SLURM job ID. The manifest is kept so
that the exact set of sessions can be audited or resubmitted.

```bash
squeue -j <job_id>
scontrol show job <job_id>
scancel <job_id>
```

The dataframe is read again by each task. The task verifies that the selected
row still has the same `session_path` recorded in the manifest, and fails
before running Suite2p if the dataframe changed or required inputs are absent.
