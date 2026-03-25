# LCSegNet: Lightweight Context-Aware Segmentation Network

## Overview

LCSegNet is a PyTorch-based semantic segmentation model designed for accurate tumor or object segmentation.  
The architecture follows an encoder-decoder paradigm enhanced with spatial refinement and feature fusion mechanisms.

The model produces:
- Pixel-wise segmentation masks  
- Boundary-aware predictions during training  



## Methodology

The network integrates:

- A ResNet34-based encoder for hierarchical feature extraction  
- Spatial Information Enhancement (SIE) modules for refining spatial representations  
- Feature Selective Fusion (FSF) blocks for multi-scale feature aggregation  
- A decoder that progressively reconstructs high-resolution segmentation outputs  

Boundary supervision is incorporated to improve edge localization and segmentation quality.



## Installation

Clone the repository and install dependencies:

```bash
git clone <your-repo-link>
cd Lcseg_net
pip install -r requirements.txt
```



## Dataset

The model expects paired image-mask data:

- Input images: RGB  
- Ground truth masks: binary (0 for background, 1 for target region)  



## Training

Run the training script:

```bash
python train.py
```

The training process optimizes a hybrid objective combining:

- Binary Cross Entropy loss  
- Dice loss  
- Boundary-aware loss  

Model selection is based on validation loss.



## Inference

Run prediction on trained weights:

```bash
python prediction.py
```

The model outputs binary segmentation masks obtained by thresholding predicted logits.



## Results Interpretation

- Lower validation loss indicates better generalization  
- A gap between training and validation loss may indicate overfitting  



## Conclusion

LCSegNet demonstrates an efficient approach to segmentation by combining spatial enhancement and feature fusion techniques within a lightweight architecture.



