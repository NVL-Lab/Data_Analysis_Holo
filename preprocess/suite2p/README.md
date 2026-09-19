# Suite2p array pipeline

This pipeline creates a manifest of selected sessions and submits one SLURM
array task per session. Each task receives its own GPU allocation and runs one
Suite2p process; sessions do not compete for a GPU inside one job.

Run from the repository root on the cluster:

```bash
module load Anaconda3/2023.07-2

python -m preprocess.suite2p.pipeline.submit \
  --dataframe /path/to/holobmi_df.parquet \
  --raw-root /data/project/nvl_lab/HoloBMI/Raw \
  --output-root /data/project/nvl_lab/processed_suite2p \
  --frame-rate 29.752 \
  --executor slurm \
  --conda-env my_suite2p_environment \
  --session-date 190930 \
  --day-index D10 \
  --mouse-id NVI12 \
  --max-parallel 2
```

`--conda-env` identifies the conda environment activated by each SLURM task.
It is required unless `SUITE2P_CONDA_ENV` is set in the submission shell:

```bash
export SUITE2P_CONDA_ENV=my_suite2p_environment
```

To run without SLURM, use the local executor. It processes selected sessions
sequentially in the current Python environment, so `--conda-env` is not
needed:

```bash
python -m preprocess.suite2p.pipeline.submit \
  --dataframe /path/to/holobmi_df.parquet \
  --raw-root /data/project/nvl_lab/HoloBMI/Raw \
  --output-root /data/project/nvl_lab/processed_suite2p \
  --frame-rate 29.752 \
  --executor local
```

For users who prefer a graphical interface, run this on a machine with a
desktop display:

```bash
python -m preprocess.suite2p.gui
```

The GUI collects the same settings as the command-line tool, including the
conda environment name, and submits the job through the current Python
environment. On a remote cluster, use X
forwarding (`ssh -X`) or run the GUI on a desktop machine that can access the
cluster and repository.

`--session-date`, `--day-index`, and `--mouse-id` are optional filters.
`--day-index`, when provided, must start with `D` (for example, `D10`).
`--max-parallel` limits the number of simultaneously running array tasks. If it
is omitted, the pipeline defaults to the number of selected sessions, so a
single filtered mouse with four matching rows will submit a four-task array.
The manifest is written under `preprocess/suite2p/manifests/`, and SLURM logs
are written under `preprocess/suite2p/logs/`. Both directories are local
working state and are ignored by Git. The command prints the manifest path and
SLURM job ID.

```bash
squeue -j <job_id>
scontrol show job <job_id>
scancel <job_id>
```

The dataframe is read again by each task. The task verifies that the selected
row still has the same `session_path` recorded in the manifest, and fails
before running Suite2p if the dataframe changed or required inputs are absent.
