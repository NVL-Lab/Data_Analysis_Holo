#!/bin/bash
#SBATCH --job-name=suite2p_holo_batch             ### Name of the job
#SBATCH --nodes=1                       ### Number of : Nodes total CPUs is equal to --cpus-per-task times --ntasks
#SBATCH --ntasks=1                     ### Number of Tasks
#SBATCH --cpus-per-task=64               ### Number of Tasks per CPU
#SBATCH --ntasks-per-socket=1          ### --ntasks-per-socket times --gres=gpu equals --ntasks
#SBATCH --gres=gpu:1                ### Number of GPUs
#SBATCH --mem=128G                        ### Memory required, 4 gigabyte
#SBATCH --partition=amperenodes-medium            ### Cheaha Partition : amperenodes
#SBATCH --time=48:00:00                 ### Estimated Time of Completion
#SBATCH --output=/home/sgurgua4/Documents/project/nvl_lab/data_analysis_holo_logs/%x_%j.out              ### Slurm Output file, %x is job name, %j is job i
#SBATCH --error=/home/sgurgua4/Documents/project/nvl_lab/data_analysis_holo_logs/%x_%j.err               ### Slurm Error file, %x is job name, %j is job id

SECONDS=0

### Loading the required CUDA and cuDNN modules
module load CUDA/12.2.0
module load cuDNN/8.9.2.26-CUDA-12.2.0

### Loading Anaconda3 module to activate `pytools-env` conda environment
module load Anaconda3/2023.07-2
conda activate suite2p_beta1

### Allocating variables
#df_dir="/home/sgurgua4/Documents/project/nvl_lab/Data_Analysis_Holo/holobmi_df.parquet"
#folder_save="/data/project/nvl_lab/HoloBMI/"
#folder_raw="/data/project/nvl_lab/HoloBMI/Raw"
#default_path=""
#frame_rate=29.752
indexes=("${@:6}")
### Runs the script in parallel
for i in "${indexes[@]}"; do
  #srun --nodes=1 --ntasks=1 python /home/sgurgua4/Documents/project/nvl_lab/Data_Analysis_Holo/preprocess/run_preprocess_sessions.py "$i" "$df_dir" "$folder_save" "$folder_raw" "$default_path" "$frame_rate" &
  srun --nodes=1 --ntasks=1 python /home/sgurgua4/Documents/project/nvl_lab/Data_Analysis_Holo/preprocess/run_preprocess_sessions2.py "$i" "$1" "$2" "$3" "$4" "$5" &
done
wait

duration=$SECONDS
printf "Time elapsed: %02d:%02d:%02d\n" \
    $((duration/3600)) \
    $((duration%3600/60)) \
    $((duration%60))
