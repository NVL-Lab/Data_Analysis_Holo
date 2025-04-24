# Repository for the HOLOBMI project


## Table of Contents
- [Overview](#overview)
- [File Descriptions](#file-descriptions)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository has code to convert the data acquired with a Prairie microscope and a pycontrol behavioral system to nwb
will also contain all the data analysis for the paper

## File Descriptions

| File/Folder       | Description |
|------------------|-------------|
| `conversion/`     | Directory containing code to convert the original data to a NWB type of file. |
| `preprocess/`     | Directory containing code to preprocess the data, including:  
  &nbsp;&nbsp;&nbsp;&nbsp;• `analyze_inputs.py` | Analyzes input data for preprocessing  
  &nbsp;&nbsp;&nbsp;&nbsp;• `convert_to_nwb.py` | Converts data to NWB format  
  &nbsp;&nbsp;&nbsp;&nbsp;• `dataframe_sessions.py` | Creates dataframes with all the sessions and the raw files names  
  &nbsp;&nbsp;&nbsp;&nbsp;• `get_voltage_analysis_prereqs.py` | Gets the prereq to obtain channels  
  &nbsp;&nbsp;&nbsp;&nbsp;• `run_2_suite2p.py` | Runs the second pass of Suite2p pipeline  
  &nbsp;&nbsp;&nbsp;&nbsp;• `syncronize_voltage_rec.py` | Syncs voltage and imaging recordings  
  &nbsp;&nbsp;&nbsp;&nbsp;• `voltage_analysis.py` | Analyzes voltage data |
| `utils.py`        | Utility functions used throughout the project. |
| `README.md`       | This file. Describes the project structure and purpose. |
| `License`         | License of the repository. |



## Installation

Instructions on setting up the environment (e.g., cloning, dependencies).

## Usage

Basic example of how to run the code or use the project.

## Contributing

Guidelines if you want others to help contribute to your code.

## License

Specify the license under which the code is released.
