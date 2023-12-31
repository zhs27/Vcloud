# Vcloud
This is the github page for test of Voint Cloud

# Environment 
First install anaconda or miniconda, and make sure cuda is available

Then install 
1. install `Pytorch3d` (depending on your system from [here](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md))
```bash
conda create -n mvtorchenv python=3.9
conda activate mvtorchenv
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
``` 

2. install `mvtorch` 

```bash
pip install git+https://github.com/ajhamdi/mvtorch
``` 
# Datasets
Download ModelNet40 dataset with following commands:
```bash
mkdir data
cd data
wget --no-check-certificate https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
```
Unzip the files.

Download ModelNet40-C using following link:
https://drive.google.com/drive/folders/10YeQRh92r_WdL-Dnog2zQfFr03UW4qXX
Place the modelnet40-c folder in data folder

# Trained models
As github's limitation to large files, I have to share it by google drive: 
https://drive.google.com/drive/folders/1afYQewKw7dEjVaAeOjXdI4aiyVtOoF5G?usp=drive_link

# Run the code
Activate your environment
For training process:
```bash
python classification.py
```

Move 3 pth files to train folder, run test.py
```bash
python test.py
```

