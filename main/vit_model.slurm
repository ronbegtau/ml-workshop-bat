#! /bin/sh

#SBATCH --job-name=vit-model
#SBATCH --output=/home/yandex/MLW2023/beglaubter/slurm_output/vit_model.out # redirect stdout
#SBATCH --error=/home/yandex/MLW2023/beglaubter/slurm_output/vit_model.err # redirect stderr
#SBATCH --partition=studentkillable # (see resources section)
#SBATCH --time=20000 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=4 # number of machines
#SBATCH --ntasks=4 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=4 # GPUs in total
-constraint="geforce_rtx_3090"

python3 vit_model.py