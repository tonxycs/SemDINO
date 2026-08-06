## About SemDINO: *A multi-scale bidirectional temporal model with foundation priors, anti-pseudo-change capability and outstanding cross-dataset performance!*

###  This repository is the official implementation:
Our papers: 
Preprint Version: [https://arxiv.org/pdf/2606.09772]

Our Architecture:

<img width="920" height="399" alt="image" src="https://github.com/user-attachments/assets/0e85a9e9-3819-4a6b-98b6-17c79c09adac" />

## 🛠️ Environment Setup (Refer to https://github.com/tonxycs/M4Fuse#pip)

#### Environment
```txt
conda create -n SemDINO python=3.8
conda activate SemDINO

# Install pytorch 
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118  --extra-index-url https://download.pytorch.org/whl/cu118

# Install other packages
pip install -r requirements.txt 

# Note: You'd better use CUDA 11.8.
```

### 🧾Dataset: (Information and Download)

#### [Information] : please refer to our datasets/RS's code and Paper's Experiments Settings Section ‌especially‌: 1.Datasets 2.Implementation Details.
#### [Download]: the following is a summary of the processed semantic change detection datasets used in this paper: 

[A] Landsat-SCD

[B] SECOND

[C] HRSCD

[https://drive.google.com/drive/folders/1DurQS_nSuU1qWm0ycvp_M4Hthe84RNFe]

How to use?  following this 🐶:

      YOUR_DATA_DIR
      ├── ...
      ├── train
      │   ├── A
      │   ├── B
      │   ├── labelA
      │   ├── labelB
      ├── val
      │   ├── A
      │   ├── B
      │   ├── labelA
      │   ├── labelB
      ├── test
      │   ├── A
      │   ├── B
      │   ├── labelA
      │   ├── labelB

### Model: ./models (more and more)

🔑 Necessary of Ours SemDINO:  [https://drive.google.com/drive/folders/1DurQS_nSuU1qWm0ycvp_M4Hthe84RNFe]  download to blocks (adapter.py: weights_path="~/blocks/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")

### Training and Test: 
1.
2.
3.
### Pred and Vis: (Follow the code above and it's very clear.)
python pred.py

## Cite SemDINO

If you find this work useful or interesting, please consider citing the following BibTeX entry.



***Stay tuned for our upcoming releases. For urgent citation, usage, or further assistance, please feel free to contact me [tonxycs@gmail.com] without any hesitation. You are very welcome!***

