# Model Card — ClothingCNN

## Overview

ClothingCNN is a compact convolutional neural network trained from scratch to classify product images into five categories: `long-sleeved`, `pants`, `shorts`, `socks` and `t-shirt`.

## Architecture

- Inverted residual and depthwise-separable convolution blocks
- SiLU activations, batch normalization and dropout
- Global average pooling and a compact classifier head
- **794,485 trainable parameters**
- Input size: **224 × 224 RGB**

## Reported final-run performance

| Metric | Value |
|---|---:|
| Test accuracy | 92.33% |
| Macro precision | 92.26% |
| Macro recall | 91.24% |
| Macro F1 | 91.68% |
| Test images | 404 |

These values come from the executed notebook output. The original checkpoint for that run was not present in the supplied files and is therefore not distributed in this repository.

## Training

- AdamW optimizer
- Cross-entropy loss
- Weighted random sampling for class imbalance
- Image augmentation
- ReduceLROnPlateau scheduler
- Early stopping on validation loss
- Maximum 70 epochs; the documented run stopped after epoch 37

## Limitations

- The model recognises only the five documented classes.
- Performance may degrade on different lighting, backgrounds or product photography styles.
- The dataset licence and original source must be verified before redistributing images.
- Reported metrics should be reproduced after creating a fresh checkpoint with the cleaned split.

## Intended use

This is an educational computer-vision project and portfolio demonstration. It is not intended for safety-critical or high-impact automated decisions.
