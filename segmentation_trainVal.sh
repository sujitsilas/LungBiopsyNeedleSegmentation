#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o joblog.SegmentationTrainVal_stratified

## Edit the line below as needed:
# GPU Model             Compute Capability   CUDA Cores   Memory
# A100                  8                    6912         80 GB   A100
# Tesla V100            7                    5120         32 GB   V100
# GeForce RTX 2080 Ti   7.5                  4352         10 GB   RTX2080Ti
# Tesla P4              6.1                  2560         8 GB    P4

#$ -l gpu,A100,cuda=2,h_rt=5:00:00 

# Email address to notify
#$ -M $sujitsilas@g.ucla.edu

# Notify when
#$ -m bea

# Load the required modules
cd /u/project/pallard/sujit009/Segmentation

. /etc/bashrc
module load conda/23.11.0
module load cuda/11.8
module load python
export PATH="{PATH}:~/.local/bin"

# Activate virtual env
conda activate /u/project/pallard/sujit009/Segmentation/my_dl_env

# Run your training script with Python
python train_val.py

# Deactivate virtual environment upon completion
conda deactivate

