
# Creating a conda virtual environment on [Hoffman2](https://www.hoffman2.idre.ucla.edu/)

## Setup Instructions

#### 1. Clear conda cache
```bash
conda clean --all
```

#### 2. Load the conda module (adjust the module name/version as necessary)
```bash
module load conda/23.11.0
```
#### 3. Create a conda environment and activate it
```bash
conda create -y -p /u/project/pallard/sujit009/Segmentation/my_dl_env python=3.11.6
conda activate /u/project/pallard/sujit009/Segmentation/my_dl_env
```
#### 4. Install the required packages
```bash
pip install matplotlib pandas torch torchvision tqdm albumentations Pillow opencv-python
```
#### 5. Deactivate the conda environment
```bash
conda deactivate
```

### Request interactive GPUs and try running an executable bash script
#### cuda represent the number of GPUs you want. Can request up to 4 v100s on hoffman
```bash
qrsh -l gpu,A100,cuda=1,h_rt=2:00:00 -pe shared 2 
```

##### Interactive session with CPUs (if needed)
```bash
qrsh -l h_data=15G,h_rt=3:00:00,h_vmem=4G -pe shared 4
```


