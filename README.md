# Visual Question Answering (VQA) on Edge Devices

## 1. Introduction

This repository provides a hands-on tutorial, on how to design, train and evaluate **Visual Question Answering (VQA)** models for **resource-constrained edge devices**, specifically the **STM32MP2 platform**.

Models are implemented in **Keras**, optimized for **TensorFlow Lite**, and can leverage the STM32MP2 **NPU** for efficient inference.

This README is structured as follows:
...

---

## 2. Quickstart

Try inference in 3 steps:

1. Clone the repo and install dependencies

```bash
git clone https://github.com/<your-username>/vqa-stm32mp2.git
cd vqa-stm32mp2
pip install -r requirements.txt
```

2. Download a pretrained model (MFB + Attention)

```bash
wget "https://docs.google.com/uc?export=download&id=1UKBwet_hAm7OCmrRjfhFd6aKOceJWKYY" -O outputs/MFBAttention.zip
unzip outputs/MFBAttention.zip -d outputs/
```

3. Run inference on an image

```bash
python --model-dir MFBAttention --question "What is this?" --image-path ./Images/COCO_val2014_000000002006.jpg
```

---

## 3. Tutorial: VQA on Edge Devices

### 3.1 Model Architectures

The implemented models are based on the architectures proposed by [**Yu et al. (2017)**](https://openaccess.thecvf.com/content_iccv_2017/html/Yu_Multi-Modal_Factorized_Bilinear_ICCV_2017_paper.html): **MFB Baseline**, **MFB + Attention**, **MFB + CoAttention**

The original architectures were modified to make them efficient on the STM32MP2 platform and leverage hardware acceleration (designed for forward CNNs):

- **ResNet-152 → MobileNet V3 Large**
  - Lighter, optimized for edge devices.
- **LSTM → Block of Temporal Convolutional Layers (TCLs)**
  - Faster and hardware-friendly sequence modeling.
- **MFB module simplification**
  - Removed power normalization
  - Sum pooling → average pooling
  - Parameters: `k = 5`, `o = 1024`

Additional improvements:

- **Word embeddings**: concatenation of **pretrained GloVe embeddings** and **learned embeddings**
  - Introduced to improve semantic representation and boost model performance.

![MFBCoAttention Architecture](Images/MFBCoAttention.png)

_Figure 2. Optimized **MFB with Co-Attention network**. The **MFB with Attention** and **MFB Baseline** variants are derived by removing, respectively, the question attention block, and both the question and image attention blocks along with the first MFB module._

The folder `models/` contains the file `vqa_models.py`, which defines the implemented model architectures:

- `MFB_Baseline(...)`: Keras implementation of the modified **MFB Baseline** model.
- `MFB_Attention(...)`: Keras implementation of the modified **MFB + Attention** model.
- `MFB_CoAttention(...)`: Keras implementation of the modified **MFB + CoAttention** model.

Each function returns a fully built **Keras model** (`tf.keras.Model`), with layers and parameters defined according to the provided configuration. While the theoretical section provides high-level block diagrams of the architectures, these functions show the **exact Keras implementation** of each model.

```bash
from models.vqa_models import MFB_Attention
model = MFB_Attention()
model.summary()
```

### 3.2 Dataset and Preprocessing

Dataset download: [VQAv2 official website](https://visualqa.org/download.html) (files to download: Balanced Real Images training and validation annotations, questions and images). Place the images and annotation files in a folder like:

```
data/vqa_dataset/
    ├── train2014/
    ├── val2014/
    ├── v2_mscoco_train2014_annotations.json
    ├── v2_mscoco_val2014_annotations.json
    ├── v2_OpenEnded_mscoco_train2014_questions.json
    └── v2_OpenEnded_mscoco_val2014_questions.json
```

Each sample contains:

- An image
- A natural language question
- Ten human-provided answers + one ground truth answer

| Set        | Questions | Images |
| ---------- | --------- | ------ |
| Training   | 443,757   | 82,783 |
| Validation | 214,354   | 40,504 |

Preprocessing:

- Answers: normalized (official guidelines) and reduced to **top 1000 most frequent**
  Training set reduced to ~87.5% of original size, and used for training. As a result, the models can only predict one of these 1000 possible answers.
- Images: Resized to **224×224** and normalized to range **[-1, 1]** (MobileNet input format).
- Questions: tokenized and padded/truncated to **15 tokens**
  Vocabulary: **6415 words** (words appearing ≥5 times).

The folder `data/` contains everything related to dataset handling and preprocessing.

- **Dataset loading**
  The function `get_vqav2(...)` in `dataset.py` loads the official VQAv2 JSON files into **Pandas DataFrames** and applies **answer normalization** (as described in the theoretical section).

  ```bash
  from data.vqa_dataset import get_vqav2
  from pathlib import Path

  split = "train2014" # ("val2014" to get the validation set)
  dataset_path = Path("data/vqa_dataset")
  df_train = get_vqav2(dataset_path, split="train2014")
  ```

- **Data generators**
  Defined in `custom_generators.py`. They are implemented as subclasses of `keras.utils.Sequence`, so they can be used directly with `model.fit()`. They handle the entire preprocessing pipeline (loading and processing images, question tokenization, etc.) and supply batches ready for training/evaluation.

### 3.3 Training procedure

con o senza KD, con o senza glove. Poi codice per addestrare

### 3.4 Evaluation and Results

... (ovviamente noi ottenuti cin KD)

### 3.5 Deployment Analysis

e TFLite conversion

---

## 4. Usage Guide

Come quella che avevo scritto prima, ma molto più riassunta.

---

## 5. Cambiare file config per avere modello diverso

...

---

## References

- Z. Yu, J. Yu, J. Fan, and D. Tao, **"Multi-modal factorized bilinear pooling with co-attention learning for visual question answering,"** in _Proceedings of the IEEE international conference on computer vision_, 2017
- S. Antol, A. Agrawal, J. Lu, M. Mitchell, D. Batra, C. L. Zitnick, and D. Parikh, **“Vqa: Visual question answering,”** in _Proceedings of the IEEE international conference on computer vision_, 2015,
- Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh, **“Making the v in vqa matter: Elevating the role of image understanding in visual question answering,”** in _Proceedings of the IEEE conference on computer vision and pattern recognition_, 2017
- W. Wang, H. Bao, L. Dong, J. Bjorck, Z. Peng, Q. Liu, K. Aggarwal, O. K. Mohammed, S. Singhal, S. Som, and F. Wei, **“Image as a foreign language: BEiT pretraining for vision and vision-language tasks,”** in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, 2023.
- G. Hinton, O. Vinyals, and J. Dean, **“Distilling the knowledge in a neural network,”** 2015.
