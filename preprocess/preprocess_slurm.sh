#!/bin/bash
#SBATCH --job-name=suite2p_holo_batch             ### Name of the job
#SBATCH --nodes=1                       ### Number of Nodes
#SBATCH --ntasks=1                     ### Number of Tasks
#SBATCH --cpus-per-task=1               ### Number of Tasks per CPU
#SBATCH --mem=96G                        ### Memory required, 4 gigabyte
#SBATCH --partition=medium            ### Cheaha Partition
#SBATCH --time=48:00:00                 ### Estimated Time of Completion
#SBATCH --output=results/%x_%j.out              ### Slurm Output file, %x is job name, %j is job i
#SBATCH --error=results/%x_%j.err               ### Slurm Error file, %x is job name, %j is job id

### Loading Anaconda3 module to activate `pytools-env` conda environment
module load Anaconda3/2023.07-2
conda activate rois

### Runs the script in parallel
for i in {0..$1}
do
  srun --nodes=1 --ntasks=1 python -c "from run_preprocess_sessions import process_single_session; process_single_session('$1','$2','$3','$4','$5','$6')" &
done
wait