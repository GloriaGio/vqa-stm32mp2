# Visual Question Answering (VQA) on Edge Devices

## 1. Introduction

This repository provides a hands-on tutorial, on how to design, train and evaluate **Visual Question Answering (VQA)** models for **resource-constrained edge devices**, specifically the **STM32MP2 platform**.

**VQA** is a task at the intersection of Computer Vision and Natural Language Processing: given an **image** and a **natural language question** about it, the model must generate the correct **answer**. VQA has many real-world applications, such as accessibility tools for visually impaired users and smart assistants that understand visual content. However, most state-of-the-art VQA models are **large and resource-hungry**, making them unsuitable for **edge devices**.

Models in this tutorial are implemented in **Keras**, optimized for **TensorFlow Lite**, and can leverage the STM32MP2 **NPU** for efficient inference.

This README is structured as follows:
...**da fare**

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

Each function returns a fully built **Keras model** (`tf.keras.Model`), with layers and parameters defined according to the provided configuration. While the theoretical section provides a high-level block diagram of the architectures, these functions show the **exact Keras implementation** of each model.

```bash
from models.vqa_models import MFB_Attention
model = MFB_Attention()
model.summary()
```

### 3.2 Dataset and Preprocessing

Dataset: **VQAv2 dataset** ([Goyal et al., 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Goyal_Making_the_v_CVPR_2017_paper.html)). See Section 4. for download from the [VQAv2 official website](https://visualqa.org/download.html).

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
from data.vqa_dataset import get_vqav2
from pathlib import Path

split = "train2014" # ("val2014" to get the validation set)
dataset_path = Path("data/vqa_dataset")
df_train = get_vqav2(dataset_path, split="train2014")
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

If you want to try or use the models already trained in this tutorial instead of training yourself, download the files from [this Drive link](https://drive.google.com/drive/folders/1iK-X6BriZnWhiYlnYmqM-lG5ooZXPkES?usp=drive_link):

MFB Baseline:

```bash
wget "https://docs.google.com/uc?export=download&id=1Ivj29hy3jt_7cH3Vs35f4s_Xp-p-Zzdw" -O outputs/MFBBaseline.zip
unzip outputs/MFBBaseline.zip -d outputs/
```

MFB + Attention:

```bash
wget "https://docs.google.com/uc?export=download&id=1UKBwet_hAm7OCmrRjfhFd6aKOceJWKYY" -O outputs/MFBAttention.zip
unzip outputs/MFBAttention.zip -d outputs/
```

MFB + CoAttention:

```bash
wget "https://docs.google.com/uc?export=download&id=1VAcbNO1LiWoQ9AScALoQFqZiSIeJ0czP" -O outputs/MFBCoAttention.zip
unzip outputs/MFBCoAttention.zip -d outputs/
```

### 4.2 Data Setup and External Resources

Before running the code for training or evaluation, make sure to download and place the required external resources in the appropriate folders:

1. **VQAv2 Dataset**

   Download the dataset from the [VQAv2 official website](https://visualqa.org/download.html).

   Training and validation questions:

```bash
wget "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip" -O data/v2_Questions_Train_mscoco.zip
unzip data/v2_Questions_Train_mscoco.zip -d data/
wget "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip" -O data/v2_Questions_Val_mscoco.zip
unzip data/v2_Questions_Val_mscoco.zip -d data/
```

Training and validation annotations:

```bash
wget "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip" -O data/v2_Annotations_Train_mscoco.zip
unzip data/v2_Annotations_Train_mscoco.zip -d data/
wget "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip" -O data/v2_Annotations_Val_mscoco.zip
unzip data/v2_Annotations_Val_mscoco.zip -d data/
```

Training and validation images:

```bash
wget "http://images.cocodataset.org/zips/train2014.zip" -O data/train2014.zip
unzip data/train2014.zip -d data/
```

```bash
wget "http://images.cocodataset.org/zips/val2014.zip" -O data/val2014.zip
unzip data/val2014.zip -d data/
```

Once downloaded, the files and images should be organized as follows:

```
data/vqa_dataset/
    ├── train2014/
    ├── val2014/
    ├── v2_mscoco_train2014_annotations.json
    ├── v2_mscoco_val2014_annotations.json
    ├── v2_OpenEnded_mscoco_train2014_questions.json
    └── v2_OpenEnded_mscoco_val2014_questions.json
```

2. **GloVe embeddings**

Download GloVe embeddings (i.e., `glove.6B.zip`) from [GloVe site](https://nlp.stanford.edu/projects/glove/).

```bash
wget "https://nlp.stanford.edu/data/glove.6B.zip" -O data/glove.6B.zip
unzip data/glove.6B.zip -d data/glove.6B/
```

**Optional:** you can still train the model without GloVe embeddings by adjusting the configuration file accordingly (see Section 5.).

3. **Teacher model logits (for Knowledge Distillation)**

Precomputed logits from the BEiT-3 teacher model (or any other teacher model) are required if you want to train using knowledge distillation (KD).

Save them in a folder structure like this:

```
data/teacher_logits/
    ├── answer2label.txt
    ├── train_logits/
    │   ├── question_id.json
    │   └── ...
    └── val_logits/
        ├── question_id.json
        └── ...
```

- `answer2label.txt` contains one dictionary per line with keys `answer` and `label`. It maps each possible teacher answer to its corresponding label, which indicates the index of the answer in the logits output of the teacher model.
- Each `question_id.json` file contains a dictionary with question-related information (question ID, image ID, ground truth answer, model answer, etc.) and a `logits` key containing the teacher model output logits.

- **Optional:** If teacher logits are not available, **you can still train the model from scratch** (see Section 3.3).

### 3.3 Training a model

1. Train using knowledge distillation:

```bash
python main_train.py --model-arch MFBBaseline --batch-size 32
```

Use `--model-arch` to specify the architecture to use: MFBBaseline, MFBAttention, or MFBCoAttention. Change the batch size with `--batch-size`.

2. Training from scratch:

```bash
python main_train.py --model-arch MFBBaseline --batch-size 32 --disable-KD
```

At the end of the training, a new folder will be created in `outputs` folder containing:

- the **trained model** saved in `.keras` format,
- the **configuration file** used for training,
- a JSON file with **preliminary performance metrics** (training/validation standard accuracy and loss on filtered dataset).

### 3.4 Evaluation

```bash
python main_eval.py --model-dir MFBBaseline --split val2014 --batch-size 32
```

where:

- `--model-dir` specifies the folder containing the trained model (created during training),
- `--split` defines which dataset split to evaluate on (`train2014` or `val2014`),
- `--batch-size` sets the evaluation batch size (if not specified, the value from the training configuration will be used).

After evaluation, the code will save:

- a JSON file with the evaluation metrics (accuracy per question type and overall accuracy)
- a JSON file containing each question ID along with the answer predicted by the model.

### 3.5 Inference

The script processes the question and image through the selected VQA model and prints the predicted answer to the console.

```bash
python inference.py --model-dir MFBBaseline --question "What is this?" --image-path ...data/vqa_dataset/val2014/COCO_val2014_000000002006.jpg
```

where:

- `--model-dir` specifies the folder containing the trained model,
- `--question` is the natural language question to ask about the image,
- `--image-path` is the path to the image file.

### 3.6 TFLite Conversion

The script converts the Keras model to TFLite format, applying per-tensor quantization. The resulting .tflite model is saved in the same folder as the original model.

```bash
python convert.py --model-dir MFBBaseline
```

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
