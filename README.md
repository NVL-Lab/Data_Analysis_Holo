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

