#!/bin/bash
#SBATCH -A cs551
#SBATCH -p academic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --job-name=DS551_DQN_Project3

# load modules or set up conda
module load anaconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# ensure you created and configured 'myenv' per Turing instructions
cd $SLURM_SUBMIT_DIR
python train_script.py --frames 5000000 --save my_dqn_breakout.pth
