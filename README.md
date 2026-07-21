## About SemDINO: *A multi-scale bidirectional temporal model with foundation priors, anti-pseudo-change capability and outstanding cross-dataset performance!*

###  This repository is the official implementation:
Our papers: 
Preprint Version: [https://arxiv.org/pdf/2606.09772]

## 📄 Abstract:
Semantic change detection (SCD) in remote sensing aims to identify land-cover transitions between bi-temporal observations while suppressing pseudo-changes caused by illumination variations, seasonal differences, and registration errors. Although Vision Foundation Models (VFMs) provide transferable semantic priors, their application to SCD remains challenging due to the mismatch between foundation-model representations and task-specific spatial features, as well as temporal-order sensitivity. To address these issues, this paper proposes SemDINO, an alignment-driven framework that integrates DINOv3 priors with hierarchical convolutional representations for cross-temporal semantic reasoning. Specifically, a Gated Pyramid Fusion (PyFu) module is developed to adaptively combine foundation-model semantics with CNN spatial details while reducing domain noise. A Multi-scale Temporal Bi-directional Transformer (M-TBTT) is introduced to achieve symmetric cross-temporal feature interaction and alleviate temporal-order bias. Furthermore, a Feature Change Enhancement (FeaCE) flow is designed to refine aligned representations and distinguish genuine semantic transitions from pseudo variations. Finally, a multi-branch decoupled prediction head jointly generates change masks, bi-temporal semantic maps, and edge constraints. Extensive experiments across five benchmark datasets demonstrate that SemDINO consistently outperforms state-of-the-art methods on both semantic and binary change detection tasks. The results validate the effectiveness of alignment-oriented representation learning for robust remote sensing change analysis.

Our Architecture: <img width="920" height="399" alt="image" src="https://github.com/user-attachments/assets/0e85a9e9-3819-4a6b-98b6-17c79c09adac" />


### Dataset: (Messages and Download)

[A] Landsat-SCD: 

[B] SECOND:

[C] HRSCD:

https://drive.google.com/file/d/14go5xbmn3uo5Gp5L-OKUuVuiLVvthJca/view?usp=drive_link download to blocks (adapter.py: weights_path="~/blocks/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")
# Stay tuned for our upcoming releases：/  If you’re interested, please feel free to contact me at [tonxycs@gmail.com]
