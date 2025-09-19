# Visual Question Answering (VQA) on Edge Devices

## 1. Introduction

This repository provides a hands-on tutorial, on how to design, train and evaluate **Visual Question Answering (VQA)** models for **resource-constrained edge devices**, specifically the **STM32MP2 platform**.

**VQA** is a task at the intersection of Computer Vision and Natural Language Processing: given an **image** and a **natural language question** about it, the model must generate the correct **answer**. VQA has many real-world applications, such as accessibility tools for visually impaired users and smart assistants that understand visual content. However, most state-of-the-art VQA models are **large and resource-hungry**, making them unsuitable for **edge devices**.

Models in this tutorial are implemented in **Keras**, optimized for **TensorFlow Lite**, and can leverage the STM32MP2 **NPU** for efficient inference.

### Requirements

- **Python 3.9**
- Other dependencies are listed in [`requirements.txt`](./requirements.txt)

---

This README is structured as follows:

- [**Section 2**](#2-quickstart): Quickstart
- [**Section 3**](#3-tutorial-vqa-on-edge-devices): Tutorial (models, dataset, training)
- [**Section 4**](#4-usage-guide): Usage Guide
- [**Section 5**](#5-configuration-and-advanced-options): Configuration and Advanced Options

---

## 2. Quickstart

Try inference in 3 steps:

1. Clone the repo and install dependencies

```bash
git clone https://github.com/GloriaGio/vqa-stm32mp2.git
cd vqa-stm32mp2
pip install -r requirements.txt
```

2. Download a pretrained model (MFB + Attention)

```bash
gdown "https://docs.google.com/uc?export=download&id=1UKBwet_hAm7OCmrRjfhFd6aKOceJWKYY" -O outputs/MFBAttention.zip
unzip outputs/MFBAttention.zip -d outputs/
```

3. Run inference on an image

```bash
python inference.py --model-dir MFBAttention --question "What is this?" --image-path ./Images/COCO_val2014_000000002006.jpg
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

Each function returns a fully built **Keras model** (`tf.keras.Model`), with layers and parameters defined according to the provided configuration. While the theoretical section provides a high-level block diagram of the architectures, these functions show the **exact Keras implementation** of each model.

```bash
from models.vqa_models import MFB_Attention
model = MFB_Attention()
model.summary()
```

### 3.2 Dataset and Preprocessing

Dataset: **VQAv2 dataset** ([Goyal et al., 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Goyal_Making_the_v_CVPR_2017_paper.html)). See [Section 4.2](#42-data-setup-and-external-resources). for download from the [VQAv2 official website](https://visualqa.org/download.html).

Each sample contains:

- An image
- A natural language question
- Ten human-provided answers + one ground truth answer

| Set        | Questions | Images |
| ---------- | --------- | ------ |
| Training   | 443,757   | 82,783 |
| Validation | 214,354   | 40,504 |

Preprocessing:

- Answers: normalized (official guidelines) and reduced to **top 1000 most frequent**.
  Training set reduced to ~87.5% of original size, and used for training. As a result, the models can only predict one of these 1000 possible answers.
- Images: Resized to **224×224** and normalized to range **[-1, 1]** (MobileNet input format).
- Questions: tokenized and padded/truncated to **15 tokens**.
  Vocabulary: **6415 words** (words appearing ≥5 times).

The folder `data/` contains everything related to dataset handling and preprocessing.

- **Dataset loading**
  The function `get_vqav2(...)` in `dataset.py` loads the official VQAv2 JSON files into **Pandas DataFrames** and applies **answer normalization**.

```python
from data.dataset import get_vqav2
from pathlib import Path

split = "train2014" # ("val2014" to get the validation set)
dataset_path = Path("vqa_dataset")
df_train = get_vqav2(dataset_path, split=split)
```

- **Data generators**
  Defined in `custom_generators.py`. They are implemented as subclasses of `keras.utils.Sequence`, so they can be used directly with `model.fit()`. They handle the entire preprocessing pipeline and supply batches ready for training/evaluation.

### 3.3 Training procedure

con o senza KD, con o senza glove. Poi codice per addestrare

### 3.4 Evaluation and Results

... (ovviamente noi ottenuti cin KD)

### 3.5 Deployment Analysis

e TFLite conversion

---

## 4. Usage Guide

### 4.1 Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/GloriaGio/vqa-stm32mp2.git
cd vqa-stm32mp2
pip install -r requirements.txt
```

#### Using the provided trained models

If you want to try the models already trained in this tutorial instead of training them yourself, download them from [this Drive link](https://drive.google.com/drive/folders/1iK-X6BriZnWhiYlnYmqM-lG5ooZXPkES?usp=drive_link):

**MFB Baseline**

```bash
gdown "https://docs.google.com/uc?export=download&id=1Ivj29hy3jt_7cH3Vs35f4s_Xp-p-Zzdw" -O outputs/MFBBaseline.zip
unzip outputs/MFBBaseline.zip -d outputs/
```

**MFB + Attention**

```bash
gdown "https://docs.google.com/uc?export=download&id=1UKBwet_hAm7OCmrRjfhFd6aKOceJWKYY" -O outputs/MFBAttention.zip
unzip outputs/MFBAttention.zip -d outputs/
```

**MFB + CoAttention**

```bash
gdown "https://docs.google.com/uc?export=download&id=1VAcbNO1LiWoQ9AScALoQFqZiSIeJ0czP" -O outputs/MFBCoAttention.zip
unzip outputs/MFBCoAttention.zip -d outputs/
```

### 4.2 Data Setup and External Resources

Before running training or evaluation, download and place the required external resources in the appropriate folders.

#### 1. VQAv2 Dataset

Download the dataset from the [VQAv2 official website](https://visualqa.org/download.html).

**Questions**

```bash
wget "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip" -O vqa_dataset/v2_Questions_Train_mscoco.zip
unzip vqa_dataset/v2_Questions_Train_mscoco.zip -d vqa_dataset/
wget "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip" -O vqa_dataset/v2_Questions_Val_mscoco.zip
unzip vqa_dataset/v2_Questions_Val_mscoco.zip -d vqa_dataset/
```

**Annotations**

```bash
wget "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip" -O vqa_dataset/v2_Annotations_Train_mscoco.zip
unzip vqa_dataset/v2_Annotations_Train_mscoco.zip -d vqa_dataset/
wget "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip" -O vqa_dataset/v2_Annotations_Val_mscoco.zip
unzip vqa_dataset/v2_Annotations_Val_mscoco.zip -d vqa_dataset/
```

**Images**

```bash
wget "http://images.cocodataset.org/zips/train2014.zip" -O vqa_dataset/train2014.zip
unzip vqa_dataset/train2014.zip -d dvqa_dataset/
```

```bash
wget "http://images.cocodataset.org/zips/val2014.zip" -O vqa_dataset/val2014.zip
unzip vqa_dataset/val2014.zip -d vqa_dataset/
```

**Final structure**

```
vqa_dataset/
    ├── train2014/
    ├── val2014/
    ├── v2_mscoco_train2014_annotations.json
    ├── v2_mscoco_val2014_annotations.json
    ├── v2_OpenEnded_mscoco_train2014_questions.json
    └── v2_OpenEnded_mscoco_val2014_questions.json
```

#### 2. GloVe Embeddings

Download GloVe embeddings (i.e., `glove.6B.zip`) from [the official site](https://nlp.stanford.edu/projects/glove/).

```bash
wget "https://nlp.stanford.edu/data/glove.6B.zip" -O resources/glove.6B.zip
unzip resources/glove.6B.zip -d resources/glove.6B/
```

**Optional:** you can still train without GloVe embeddings by adjusting the configuration file (see [Section 5](#5-configuration-and-advanced-options)).

#### 3. Teacher Model Logits (for Knowledge Distillation)

Precomputed logits from the BEiT-3 teacher model (or another teacher) are required if you want to use knowledge distillation (KD).

Expected folder structure:

```
resources/teacher_logits/
    ├── answer2label.txt
    ├── train_logits/
    │   ├── question_id.json
    │   └── ...
    └── val_logits/
        ├── question_id.json
        └── ...
```

- `answer2label.txt`: maps each possible answer to its index in the teacher’s logits.
- Each `question_id.json`: contains question metadata and a `logits` key with the teacher model outputs.

**Optional:** If teacher logits are not available, **you can still train the model from scratch** (see [Section 4.3](#43-training-a-model) or [Section 5](#5-configuration-and-advanced-options)).

### 4.3 Training a Model

**With Knowledge Distillation**

```bash
python main_train.py --model-arch MFBBaseline --batch-size 32
```

**From scratch**

```bash
python main_train.py --model-arch MFBBaseline --batch-size 32 --disable-KD
```

Arguments:

- `--model-arch`: choose the architecture (`MFBBaseline`, `MFBAttention`, or `MFBCoAttention`)
- `--batch-size`: set the batch size (default: 32).

After training, a new folder will appear in `outputs/` containing:

- the trained model (`.keras` format),
- the training configuration file,
- JSON file with preliminary performance metrics (training/validation accuracy and loss on filtered dataset).

### 4.4 Evaluation

Evaluate a trained model:

```bash
python main_eval.py --model-dir MFBBaseline --split val2014 --batch-size 32
```

Arguments:

- `--model-dir`: folder containing the trained model,
- `--split`: dataset split to evaluate (`train2014` or `val2014`),
- `--batch-size`: evaluation batch size (defaults to training value).

Outputs:

- JSON file with evaluation metrics (per-question type and overall accuracy),
- JSON file with predicted answers for each question ID.

### 4.5 Inference

Run inference on a single image + question:

```bash
python inference.py --model-dir MFBBaseline --question "What is this?" --image-path ./vqa_dataset/val2014/COCO_val2014_000000002006.jpg
```

Arguments:

- `--model-dir`: folder containing the trained model,
- `--question`: natural language question,
- `--image-path`: path to the input image.

### 4.6 TFLite Conversion

Convert a trained Keras model to TensorFlow Lite with per-tensor quantization:

```bash
python convert.py --model-dir MFBBaseline
```

The .tflite file will be saved in the same folder as the original model.

---

## 5. Configuration and Advanced Options

...

---

## References

- Z. Yu, J. Yu, J. Fan, and D. Tao, **"Multi-modal factorized bilinear pooling with co-attention learning for visual question answering,"** in _Proceedings of the IEEE international conference on computer vision_, 2017
- S. Antol, A. Agrawal, J. Lu, M. Mitchell, D. Batra, C. L. Zitnick, and D. Parikh, **“Vqa: Visual question answering,”** in _Proceedings of the IEEE international conference on computer vision_, 2015,
- Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh, **“Making the v in vqa matter: Elevating the role of image understanding in visual question answering,”** in _Proceedings of the IEEE conference on computer vision and pattern recognition_, 2017
- W. Wang, H. Bao, L. Dong, J. Bjorck, Z. Peng, Q. Liu, K. Aggarwal, O. K. Mohammed, S. Singhal, S. Som, and F. Wei, **“Image as a foreign language: BEiT pretraining for vision and vision-language tasks,”** in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, 2023.
- G. Hinton, O. Vinyals, and J. Dean, **“Distilling the knowledge in a neural network,”** 2015.
