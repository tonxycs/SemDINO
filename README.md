## About SemDINO: *A multi-scale bidirectional temporal model with foundation priors, anti-pseudo-change capability and outstanding cross-dataset performance!*

###  This repository is the official implementation:
Our papers: 
Preprint Version: [https://arxiv.org/pdf/2606.09772]

## 📄 Abstract:
Semantic change detection (SCD) in remote sensing aims to identify land-cover transitions between bi-temporal observations while suppressing pseudo-changes caused by illumination variations, seasonal differences, and registration errors. Although Vision Foundation Models (VFMs) provide transferable semantic priors, their application to SCD remains challenging due to the mismatch between foundation-model representations and task-specific spatial features, as well as temporal-order sensitivity. To address these issues, this paper proposes SemDINO, an alignment-driven framework that integrates DINOv3 priors with hierarchical convolutional representations for cross-temporal semantic reasoning. Specifically, a Gated Pyramid Fusion (PyFu) module is developed to adaptively combine foundation-model semantics with CNN spatial details while reducing domain noise. A Multi-scale Temporal Bi-directional Transformer (M-TBTT) is introduced to achieve symmetric cross-temporal feature interaction and alleviate temporal-order bias. Furthermore, a Feature Change Enhancement (FeaCE) flow is designed to refine aligned representations and distinguish genuine semantic transitions from pseudo variations. Finally, a multi-branch decoupled prediction head jointly generates change masks, bi-temporal semantic maps, and edge constraints. Extensive experiments across five benchmark datasets demonstrate that SemDINO consistently outperforms state-of-the-art methods on both semantic and binary change detection tasks. The results validate the effectiveness of alignment-oriented representation learning for robust remote sensing change analysis.

Our Architecture:

<img width="920" height="399" alt="image" src="https://github.com/user-attachments/assets/0e85a9e9-3819-4a6b-98b6-17c79c09adac" />


### Dataset: (Information and Download)

#### [Information] : please refer to our datasets/RS's code and Paper's Experiments Settings Section ‌especially‌: 1.Datasets 2.Implementation Details.
#### [Download]: the following is a summary of the processed semantic change detection datasets used in this paper: 

[A] Landsat-SCD

[B] SECOND

[C] HRSCD

[https://drive.google.com/drive/folders/1DurQS_nSuU1qWm0ycvp_M4Hthe84RNFe]

How to use?  following this:

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

### Training and Test: (Information and Download)

### Pred and Vis: (Follow the code above and it's very clear.)
python pred.py

## Cite SemDINO

If you find this work useful or interesting, please consider citing the following BibTeX entry.

```
@misc{tong2026semdinodinov3guidedcrosstemporalsemantic,
      title={SemDINO: DINOv3-Guided Cross-Temporal Semantic Alignment Network for Remote Sensing Change Detection}, 
      author={Xinyu Tong and Meihua Zhou and Jinxiao Sun and Lei Wang},
      year={2026},
      eprint={2606.09772},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.09772}, 
}
```

***Stay tuned for our upcoming releases. For urgent citation, usage, or further assistance, please feel free to contact me [tonxycs@gmail.com] without any hesitation. You are very welcome!***

