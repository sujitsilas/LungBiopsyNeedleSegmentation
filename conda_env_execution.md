
### Creating a conda virtual environment on Hoffman2

## Clear conda cache
conda clean --all

## Load the conda module (adjust the module name/version as necessary)
module load conda/23.11.0

## Create the conda environment
conda create -y -p /u/project/pallard/sujit009/Segmentation/my_dl_env python=3.11.6

conda activate /u/project/pallard/sujit009/Segmentation/my_dl_env

## Install the required packages
pip install matplotlib pandas torch torchvision tqdm albumentations Pillow opencv-python

## Deactivate the conda environment
conda deactivate



## Request interactive GPUs and try running an executable bash script
## cuda represent the number of GPUs you want. Can request up to 4 v100s on hoffman
qrsh -l gpu,A100,cuda=1,h_rt=2:00:00 -pe shared 2 

### Interactive session (if needed)
qrsh -l h_data=15G,h_rt=3:00:00,h_vmem=4G -pe shared 4

DIRECTORY=/u/project/pallard/sujit009/Segmentation/my_dl_env

