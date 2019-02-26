# 3D-IQTT

Dataset loading scripts for the 3D IQ test task (3D-IQTT) dataset.

### Prerequisites:

This is the companion code repository for the 3D-IQTT dataset. You need to download the dataset first. No worries, you don't need to sign anything. 

The dataset consists of 3 files:

- training, 46GB compressed, 79GB uncompressed*
- test, 7.9GB
- validation, 7.9GB

The command for decompressing the training dataset is 
    
    unxz --verbose 3diqtt-v2-train.h5.xz
    
The `--verbose` is important to get a progress bar, because this takes ~1-3h.  

### Install

    git clone https://github.com/fgolemo/3D-IQTT.git
    cd 3D-IQTT
    pip install -e . # you might have to use "sudo"
   
### How to use

Check out the file `scripts/1-load-train-data.py`. That file contains a self-contained example on how to load the different datasets. You just have to change the directoy in line `10`.

### 
        
        
