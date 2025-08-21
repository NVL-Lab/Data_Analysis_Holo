# Data_Analysis_Holo

Branch: `dataframe_and_slurm`

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [TODOs](#todos)

---

## Overview

This repository provides tools and scripts for data analysis workflows in the NVL Lab.

It supports:

- Efficient dataframe-based data processing (e.g. using Pandas)
- Scalable parallel processing via SLURM job scheduler on HPC clusters
- Modular and reusable preprocessing and analysis pipeline
- Example notebook for interactive exploration

---

## Features

- Data parsing and transformation with Pandas
- Batch processing through SLURM
- Support for large datasets
- Modular and well-documented codebase
- Example configuration files and notebooks
- Easy to extend for new data formats or analysis pipelines

---


## Project Structure


| Folder / File                  | Description                                           |
|--------------------------------|------------------------------------------------------ |
| `Preprocess/`                   | Preprocessing scripts and SLURM batch preparation    |
| `Preprocess/dataframe_sessions.py` | Session-based dataframe preparation               |
| `Preprocess/prepare_batches.py` | Prepare batch jobs for processing                    |
| `Preprocess/prepare_tiff.py`    | TIFF file preparation                                |
| `Preprocess/run_suite2p.py`     | Script to run Suite2p pipeline                       |
| `Preprocess/session_paths.py`   | Manage session paths                                 |
| `Preprocess/slurm_array.sh`     | SLURM batch array job script                         |
| `FilePath.py`                   | File path utilities                                  |
| `HoloBMI sampledata.ipynb`      | Example Jupyter notebook with sample data            |
| `run_2_suite2p.py`              | Second Pass script                                   |
| `utils/`                        | Utility functions and constants                      |
| `utils/analysis_constants.py`   | Constants used in analysis                           |
| `LICENSE`                       | Project license                                      |
| `README.md`                     | Project documentation (this file)                    |


## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/NVL-Lab/Data_Analysis_Holo.git
cd Data_Analysis_Holo
git checkout dataframe_and_slurm

## USAGE
 ### Running Locally

- **Run full pipeline:**
 --First run prepare_batches.py which will produce json file containning session_paths for suite2p.
 --After specify run_suite2p script in slurm array.sh file.
 --Next run slurm.sh in cheaha which will start execution.

 
## Contributing



## License


## TODO:
 -- Add unit tests

 -- Improve SLURM script templates

 -- Add error handling and logging

 -- Define and Import constants and variable from UTILS folders.
```


# SLURM Job Scheduler Overview

These lines tell the SLURM job scheduler how to run your job.

`#!/bin/bash`: Specifies that this script should be run using the Bash shell.

`#SBATCH --job-name=suite2p_batch`: Names your job "suite2p_batch" so it’s easier to track in SLURM’s job queue.

`#SBATCH --nodes=1`: Requests 1 node (a physical machine). Most jobs only need one unless you're doing large-scale parallel computing.

`#SBATCH --ntasks=1`: Specifies that only 1 task will be run. A task is a single process.

`#SBATCH --cpus-per-task=1`: Allocates 1 CPU core for your task. You can increase this if your program can use multiple cores.

`#SBATCH --mem=512G`: Requests 512 gigabytes of RAM. Make sure you really need that much—some clusters have memory limits.

`#SBATCH --partition=medium`: Assigns the job to the "medium" queue. Partitions are like different lanes—some may be faster, slower, or have different limits.

`#SBATCH --time=49:00:00`: Sets a time limit of 49 hours. If the job runs longer, it will be stopped automatically.

`#SBATCH --array=0-9%5`: This creates an array job with 10 tasks (from index 0 to 9), but only 5 can run at the same time. Useful for batch processing.

`#SBATCH --output=logs/run1/job_%A_%a.out`: Sets where to save standard output (e.g., print statements). `%A` is the main job ID, `%a` is the array index.

`#SBATCH --error=logs/run1/job_%A_%a.err`: Sets where to save standard error messages (e.g., crashes, tracebacks).

## Job Execution

`module load Anaconda3/4.4.0`: Loads the Anaconda module (version 4.4.0) on the cluster. Modules are used to manage software environments.

`source activate suite2p_env`: Activates your Conda environment named `suite2p_env`, which should contain all the dependencies for Suite2p.

`python run_suite2p.py`: Runs your Python script. This is the main action your job performs.

---

**Note:** Pass these in the command prompt.

### `scontrol`

This one’s the control panel. You can use it to get detailed job info, cancel jobs, or manipulate jobs and nodes.

`scontrol show job <jobID>`

### `squeue`

It shows what jobs are currently in the SLURM job queue. It doesn’t show completed jobs, but it tells you what’s in progress or on hold—like an airport departures board for your computational flights.

`squeue -u <username>`
`squeue -j <jobID>`

---

I recommend using the correct version of Anaconda before submitting your jobs to SLURM and going through the SLURM documentation to submit jobs or learn more about the keywords in detail.
