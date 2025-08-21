#!/bin/bash
#SBATCH --job-name=suite2p_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512G
#SBATCH --partition=medium
#SBATCH --time=49:00:00
#SBATCH --array=0-9%5 
#SBATCH --output=logs/run1/job_%A_%a.out
#SBATCH --error=logs/run1/job_%A_%a.err

module load Anaconda3/4.4.0
source activate suite2p_env

python run_suite2p.py
