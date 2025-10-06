# Visual Question Answering (VQA) on Edge Devices

## 1. Introduction

This repository provides a hands-on tutorial that shows you how to design, train, and evaluate **Visual Question Answering (VQA)** models for **resource-constrained edge devices**, specifically the **STM32MP2 MPU**.

**VQA** is a task at the intersection of computer vision and natural language processing: given an **image** and a **natural language question** about it, the model must generate the correct **answer**. VQA has many real-world applications, such as accessibility tools for visually impaired users and smart assistants that understand visual content. However, most state-of-the-art VQA models are **large and resource-hungry**, making them unsuitable for **edge devices**.

The models in this tutorial are implemented in Keras 3.9, optimized for TensorFlow Lite, and designed to leverage the STM32MP2 NPU for efficient inference.

This README is structured as follows:

- [**Section 2**](#2-quickstart): Quickstart (inference in 3 steps)
- [**Section 3**](#3-tutorial-vqa-on-edge-devices): Tutorial (models, dataset, training and evaluation)
- [**Section 4**](#4-usage-guide): Usage Guide
- [**Section 5**](#5-performance-testing-on-stm32mp2): Performance Testing on STM32MP2
- [**Section 6**](#6-configuration-and-advanced-options): Configuration and Advanced Options

---

### Requirements

- **Python 3.9**
- Other dependencies are listed in [`requirements.txt`](./requirements.txt)

---

## 2. Quickstart

Try inference in 3 steps with the MFB + Attention pre-trained models:

1. Clone the repo and install dependencies

(Optional) create a separate Conda environment:

```bash
conda create --name tutorialVQAenv python=3.9
conda activate VQAenv
pip install --upgrade pip
```

```bash
git clone https://github.com/GloriaGio/vqa-stm32mp2.git
cd vqa-stm32mp2
pip install -r requirements.txt
mkdir vqa_dataset resources
```

2. Download a pretrained model (MFB + Attention)

```bash
gdown "https://docs.google.com/uc?export=download&id=1UKBwet_hAm7OCmrRjfhFd6aKOceJWKYY" -O outputs/MFBAttention.zip
unzip outputs/MFBAttention.zip -d outputs/
```

3. Run inference on a sample image

<p align="center">
<img src="Images/COCO_val2014_000000002006.jpg" alt="SampleImage" width="150"/>
</p>

_Figure 1. Sample image from the VQAv2 validation set._

```bash
python inference.py --model-dir MFBAttention --question "What is this?" --image-path ./Images/COCO_val2014_000000002006.jpg
```

- Expected answer: `bus`
- **Change the question:** replace the text after --question with your own question (for example: `"What color is the vehicle?"` -> `purple`, `"Which country is this?"` -> `england`, `"How many people are there?"` -> `2`).
- **Change the image:** replace the path after --image-path with the path to your desired image.

### 2.1 Repository Structure

```
vqa-stm32mp2/
├── data/                 # Data loading, preprocessing, and generators
├── Images/               # Figures for README + sample image for inference
├── models/               # Model architectures (MFB Baseline, Attention, CoAttention) and GloVe handling
├── outputs/              # Saved trained models with configs and performance logs
├── resources/            # External resources (GloVe embeddings and teacher logits)
├── train/                # Training functions (KD/from scratch, distiller, performance)
├── utils/                # Configuration management utilities
├── vqa_dataset/          # Folder for VQAv2 dataset (images, questions, annotations)
├── config.json           # Model/training configuration file
├── convert_to_tflite.py  # TFLite conversion script
├── inference.py          # Script for running inference on an image + question
├── main_eval.py          # Evaluation entry point
├── main_train.py         # Training entry point
├── requirements.txt      # Python dependencies
└── README.md
```

---

## 3. Tutorial: VQA on Edge Devices

### 3.1 Model Architectures

The implemented models are based on the architectures proposed by [**Yu et al. (2017)**](https://openaccess.thecvf.com/content_iccv_2017/html/Yu_Multi-Modal_Factorized_Bilinear_ICCV_2017_paper.html): **MFB Baseline**, **MFB + Attention**, **MFB + CoAttention**. These models use MFB (Multi-modal Factorized Bilinear pooling) to fuse visual and textual features, exploring different strategies to integrate attention.

The original architectures were modified to run efficiently on the STM32MP2 MPU, taking advantage of its NPU:

- **ResNet-152 → MobileNet V3 Large**
  - Lighter, optimized for edge devices.
- **LSTM → Block of Temporal Convolutional Layers (TCLs)**
  - Faster and hardware-friendly sequence modeling.
- **MFB module simplification**
  - Removed power normalization
  - Sum pooling → average pooling
  - Parameters: `k = 5`, `o = 1024`

Additional improvements:

- **Word embeddings**: concatenation of **pretrained GloVe embeddings** (Global Vectors for Word Representation, a method to capture semantic meaning of words; [learn more](https://nlp.stanford.edu/projects/glove/)) and **learned embeddings**
  - Introduced to improve semantic representation and boost model performance.

![MFBCoAttention Architecture](Images/MFBCoAttention.png)

_Figure 2. This figure shows the optimized **MFB with Co-Attention network**. The **MFB with Attention** variant is obtained by removing the question attention block from this architecture. The **MFB Baseline** variant is obtained by removing both the question and image attention block and the first MFB module._

The folder `models/` contains the file `vqa_models.py`, which defines the implemented model architectures:

- `MFB_Baseline(...)`: Keras implementation of the modified **MFB Baseline** model.
- `MFB_Attention(...)`: Keras implementation of the modified **MFB + Attention** model.
- `MFB_CoAttention(...)`: Keras implementation of the modified **MFB + CoAttention** model.

Each function returns a complete **Keras model** (`tf.keras.Model`), with layers and parameters defined based on the provided configuration. These functions show the **exact Keras 3.9 implementation** of each model.

To see the model summary, first start the Python interpreter in your terminal:

```bash
python
```

Then, in the Python prompt, run the following commands:

```python
from models.vqa_models import MFB_Attention
model = MFB_Attention() # Instantiate the model (pretrained MobileNetV3 Large weights will be downloaded and stored in ~/.keras/models)
model.summary() # Print the model summary
```

### 3.2 Dataset and Preprocessing

**Dataset: VQAv2 dataset ([Goyal et al., 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Goyal_Making_the_v_CVPR_2017_paper.html)).** See [Section 4.2](#42-data-setup-and-external-resources) for download from the [VQAv2 official website](https://visualqa.org/download.html).

Each sample contains:

- An image
- A natural language question
- Ten human-provided answers + one ground truth answer

| Set        | #Questions | #Images |
| ---------- | ---------- | ------- |
| Training   | 443,757    | 82,783  |
| Validation | 214,354    | 40,504  |

Preprocessing:

- **Answers**: normalized (following the official guidelines) and **reduced to the top 1000 most frequent ones**.
  - The training set is reduced to ~87.5% of original size and used for model training. As a result, the models can only predict one of these 1000 possible answers.
- **Images**: Resized to 224×224 and normalized to range [-1, 1] (MobileNet input format).
- **Questions**: tokenized and padded/truncated to **15 tokens**
  - Vocabulary: **6415 words** (words appearing ≥5 times).

The folder `data/` contains everything related to dataset handling and preprocessing.

- **Dataset loading.**
  The function `get_vqav2(...)` in `dataset.py` loads the official VQAv2 JSON files into Pandas DataFrames and applies answer normalization.

Before running the code, download the VQAv2 JSON files for training:

```bash
wget "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip" -O vqa_dataset/v2_Questions_Train_mscoco.zip
unzip vqa_dataset/v2_Questions_Train_mscoco.zip -d vqa_dataset/
wget "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip" -O vqa_dataset/v2_Annotations_Train_mscoco.zip
unzip vqa_dataset/v2_Annotations_Train_mscoco.zip -d vqa_dataset/
```

Then, start the Python interpreter in your terminal

```bash
python
```

At the Python prompt, run the following commands:

```python
from data.dataset import get_vqav2
from pathlib import Path


split = "train2014" # Load the training split
dataset_path = Path("vqa_dataset")
df_train = get_vqav2(dataset_path, split=split) # Load the dataset into a Pandas DataFrame
print(df_train.columns) # Inspect the DataFrame columns
```

- **Data generators.**
  Defined in `custom_generators.py`, they are implemented as subclasses of `keras.utils.Sequence`, so they can be used directly with `model.fit()`. They handle the entire preprocessing pipeline and supply batches ready for training/evaluation.

### 3.3 Training procedure

Two training modes are supported:

- **From scratch**
  - cross-entropy (CE) loss with ground truth labels (_y_).
- **With knowledge distillation (KD)** as described in [Hinton et al., 2015](https://arxiv.org/abs/1503.02531)
  - the loss combines two terms:
    - **Student loss**: cross-entropy (CE) with ground truth labels (_y_).
    - **Distillation loss**: Kullback–Leibler (KL) divergence between teacher logits (_t_) and student logits (_s_).
  - **Teacher model:** BEiT-3 ([Wang et al., 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Image_as_a_Foreign_Language_BEiT_Pretraining_for_Vision_and_CVPR_2023_paper.html)) fine-tuned on VQAv2 [(beit3_large_indomain_patch16_224)](https://github.com/microsoft/unilm/tree/master/beit3#fine-tuning-on-vqav2-visual-question-answering) (82.53% accuracy on test-dev, 683M parameters).
  - Parameters: `T = 3`, `α = 0.1`

<p align="center">
<img src="Images/Loss.png" alt="Loss" width="330"/>
</p>

**Class imbalance handling:** Since _yes/no_ answers represent ~40% of the dataset, **sample weights** inversely proportional to answer frequency were applied.

**Training setup:**

- Optimizer: **Adam**, learning rate `1e-4`
- Epochs: **10** (~1 hour per epoch, ~10 hours total per model on a NVIDIA GeForce GTX 1060 6 GB)

The file`train/trainer.py` provides two functions for model training:

- `train_from_scratch(...)` for training from scatch:

  - Calls `model.compile()` with **Adam optimizer** and **cross-entropy loss**.
  - Defines Keras callbacks (callbacks can be customized in this file).
  - Runs `model.fit(...)` and returns the trained model.

- `train_with_KD(...)` for training with KD:

  - Wraps the model inside a custom `Distiller(keras.Model)` class (see [Keras KD Code Examples](https://keras.io/examples/)).
  - Compiles the distiller with:
    - Optimizer: **Adam**
    - Losses: **cross-entropy** (student) + **KL divergence** (distillation)
    - KD parameters: `alpha` and `T` (from config).
  - Defines Keras callbacks (callbacks can be customized in this file).
  - Runs `distiller.fit(...)` to train the student model.
  - Returns the trained student model.

Both functions handle validation during training and return a trained model.

The entire training pipeline is handled by `main_train.py`: it loads data and reduces answers (top 1000 most frequent ones), builds custom generators, creates the model, trains from scratch or with KD, and saves the trained model along with used configuration and preliminary performance.

### 3.4 Evaluation and Results

The performance of VQA models are evaluated by comparing the model’s predicted answers to the human-provided reference answers ([Antol et al., 2015](https://openaccess.thecvf.com/content_iccv_2015/html/Antol_VQA_Visual_Question_ICCV_2015_paper.html)). Given the model’s predicted answer _a<sub>i</sub>_ for question _i_, and the ten human-provided reference answers, the accuracy of a model is computed as follows:

<p align="center">
<img src="Images/Accuracy.png" alt="Loss" width="330"/>
</p>

where _N_ is the number of questions and _count(a<sub>i</sub>)_ denotes the number of annotators who provided the answer _a<sub>i</sub>_ to question _i_.

The entire evaluation pipeline is handled by `main_eval.py`: it loads the data and the trained model, obtains model answers, computes accuracy, and saves model answers and performance.

#### VQA Performance (Overall and by Answer Type: Yes/No, Number, and Other)

| Model        | #Params | Overall (%) | Yes/No (%) | Number (%) | Other (%) |
| ------------ | ------- | ----------- | ---------- | ---------- | --------- |
| MFB Baseline | 24.4M   | 56.0        | 76.7       | 36.5       | 45.4      |
| MFB + Att.   | 40.4M   | **57.0**    | 77.7       | 37.2       | 46.5      |
| MFB + CoAtt. | 51.2M   | 56.4        | 76.9       | 36.8       | 46.1      |

Results were obtained using the knowledge distillation (KD) setup with BEiT-3 as teacher (see [Section 3.3](#33-training-procedure)).
The best-performing model is **MFB + Attention**, which was then deployed on the STM32MP2 MPU.

### 3.5 Deployment Analysis

| Model      | Execution Device | Inference time (ms) | Power (W) |
| ---------- | ---------------- | ------------------- | --------- |
| MFB + Att. | GPU/NPU          | **56**              | 0.75      |
| MFB + Att. | CPU              | 434                 | 0.80      |

These results show that **MFB + Attention** runs efficiently on the STM32MP2, with significantly faster execution and lower power consumption when using the GPU/NPU compared to CPU execution.

---

## 4. Usage Guide

### 4.1 Installation

(Optional) create a separate Conda environment:

```bash
conda create --name tutorialVQAenv python=3.9
conda activate VQAenv
pip install --upgrade pip
```

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/GloriaGio/vqa-stm32mp2.git
cd vqa-stm32mp2
pip install -r requirements.txt
mkdir vqa_dataset resources
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

Download the VQAv2 dataset from the [official website](https://visualqa.org/download.html). The files will be stored in the `vqa_dataset` folder:

- **Questions** (training and validation):

```bash
wget "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip" -O vqa_dataset/v2_Questions_Train_mscoco.zip
unzip vqa_dataset/v2_Questions_Train_mscoco.zip -d vqa_dataset/
wget "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip" -O vqa_dataset/v2_Questions_Val_mscoco.zip
unzip vqa_dataset/v2_Questions_Val_mscoco.zip -d vqa_dataset/
```

- **Annotations** (training and validation):

```bash
wget "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip" -O vqa_dataset/v2_Annotations_Train_mscoco.zip
unzip vqa_dataset/v2_Annotations_Train_mscoco.zip -d vqa_dataset/
wget "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip" -O vqa_dataset/v2_Annotations_Val_mscoco.zip
unzip vqa_dataset/v2_Annotations_Val_mscoco.zip -d vqa_dataset/
```

- **Images**

⚠️ Training images are large (~13.5 GB)

```bash
wget "http://images.cocodataset.org/zips/train2014.zip" -O vqa_dataset/train2014.zip
unzip vqa_dataset/train2014.zip -d vqa_dataset/
```

⚠️ Validation images are large (~6.6 GB)

```bash
wget "http://images.cocodataset.org/zips/val2014.zip" -O vqa_dataset/val2014.zip
unzip vqa_dataset/val2014.zip -d vqa_dataset/
```

**Folder structure after extraction:**

```
vqa-stm32mp2/vqa_dataset/
             ├── train2014/
             ├── val2014/
             ├── v2_mscoco_train2014_annotations.json
             ├── v2_mscoco_val2014_annotations.json
             ├── v2_OpenEnded_mscoco_train2014_questions.json
             └── v2_OpenEnded_mscoco_val2014_questions.json
```

#### 2. GloVe Embeddings

Download GloVe embeddings (i.e., `glove.6B.zip`) from [the official website](https://nlp.stanford.edu/projects/glove/):

```bash
wget "https://nlp.stanford.edu/data/glove.6B.zip" -O resources/glove.6B.zip
unzip resources/glove.6B.zip -d resources/glove.6B/
```

**Optional:** you can still train without GloVe embeddings by adjusting the configuration file (see [Section 5](#5-configuration-and-advanced-options)).

#### 3. Teacher Model Logits (for Knowledge Distillation)

Precomputed logits from the BEiT-3 teacher model (or another teacher) are required if you want to use knowledge distillation (KD).

Expected folder structure:

```
vqa-stm32mp2/resources/teacher_logits/
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

⚠️ Make sure you download the VQAv2 dataset before training.

**With Knowledge Distillation**

```bash
python main_train.py --model-arch MFBBaseline --num-epochs 10 --batch-size 32
```

**From scratch**

➡️ Use this command if you do NOT have teacher logits!

```bash
python main_train.py --model-arch MFBBaseline --disable-KD --num-epochs 10  --batch-size 32
```

Arguments:

- `--model-arch`: choose the architecture (`MFBBaseline`, `MFBAttention`, or `MFBCoAttention`)
- `--num-epochs`: number of training epochs (default: 10)
- `--batch-size`: batch size for training (default: 32)

After training, a new folder will appear in `outputs/`. Its name indicates the model architecture used and the date and time when training was started (to avoid overwriting previous runs). The folder contains:

- the trained model (`.keras` format),
- the training configuration file,
- JSON file with preliminary performance metrics (training/validation accuracy and loss on filtered dataset).

### 4.4 Evaluation

⚠️ Make sure you download the VQAv2 dataset before evaluation.

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

⚠️ Make sure you download the VQAv2 dataset before conversion.

Convert a trained Keras model to TensorFlow Lite with per-tensor quantization:

```bash
python convert_to_tflite.py --model-dir MFBBaseline
```

The .tflite model will be saved in the same folder as the original model.

---

## 5. Performance Testing on STM32MP2

You can test and benchmark the trained VQA models on the STM32MP2 MPU via the [ST Edge AI Developer Cloud](https://stedgeai-dc.st.com/home), which lets you execute models on real hardware without needing a physical board.

Steps:

1. **Upload your model**
   - Use the `.tflite` model generated in [Section 4.6](#46-tflite-conversion).
   - Place it in the Developer Cloud workspace and select **Start**.
2. **Select the STM32 MPUs platform**
   - Skip the quantization step.
3. **Enable optimization**
   - Select the **Optimize** option to leverage the NPU and improve inference time.
4. **Start Benchmarking on STM32MP257F-EV1**
   - Make sure to use the NPU selecting it in advanced setting ⚙️
   - Record the inference time for your model.
   - Compare performance using different configurations: NPU, 1 CPU core, 2 CPU cores

---

## 6. Configuration and Advanced Options

All training and evaluation settings are stored in the [`config.json`](config.json) file located in the root of the repository.
This file is divided into three sections: model, training, and paths.

By default, the training scripts load parameters from `config.json`.

➡️ If you start training with the provided configuration file (and use KD as indicated), you will reproduce the same models and results described in this tutorial.

➡️ By editing `config.json`, you can easily experiment with different architectures, hyperparameters, or resources to build your own custom VQA models.

⚠️ Command-line arguments (e.g. `--model-arch`, `--epochs`) will override the corresponding values in `config.json`.

### 6.1 Model Parameters

- `model_architecture`: VQA model architecture. Options: `MFBBaseline`, `MFBAttention`, `MFBCoAttention`.
- `max_length`: sequence length for tokenized questions (padded or truncated to `max_length`).
- `num_vocab_words`: size of the question vocabulary. If set to `-1`, vocabulary size is determined by `min_frequency`.
- `min_frequency`: minimum frequency for a word to be included in the vocabulary (ignored if `num_vocab_words` > 0).
- `image_size`: input image size (e.g. `224` → `224×224`).
- `num_channels`: number of image channels (`3` = RGB, `1` = grayscale).
- `num_classes`: number of possible answers (the model classifies among this set).
- `consider_teacher`: (`true`/`false`) if `true`, the answer space is built by intersecting the most frequent answers in the training set with those available in the teacher model. The final number of answers is the minimum between `num_classes` and the teacher’s available answers. If `false`, only frequency in the training set is used.
  - `true` + `knowledge_distillation=true`: train with KD.
  - `true` + `knowledge_distillation=false`: train from scratch, but on the teacher’s answer space.
  - `false`: standard training from scratch.
- `k_window`: hyperparameter _k_ of the MFB module.
- `output_MFB`: hyperparameter _o_ of the MFB module.
- `num_attention_glimps`: number of attention glimpses (concatenated after Global Avg Pooling and before MFB module).
- `use_glove`: (`true`/`false`) whether to combine GloVe embeddings with learned embeddings. Set to `false` if you don't want to use GloVe embeddings.
- `embedding_dim`: dimension of the word embeddings (applies to both GloVe and learned embeddings).
- `dropout_rate`: dropout rate applied during training.

### 6.2 Training parameters

- `num_epochs`: number of training epochs.
- `lr`: learning rate.
- `batch_size`: batch size for training.
- `alpha`: KD balancing coefficient (α).
- `temperature`: KD softmax temperature (T).
- `knowledge_distillation`: (`true`/`false`) whether to use KD during training.
  - if `consider_teacher=false`, this is automatically disabled.
- `restore_best_weights`: (`true`/`false`) whether to restore the best model (based on validation loss) or keep the final epoch model.

---

## References

- Z. Yu, J. Yu, J. Fan, and D. Tao, **"Multi-modal factorized bilinear pooling with co-attention learning for visual question answering,"** in _Proceedings of the IEEE international conference on computer vision_, 2017
- S. Antol, A. Agrawal, J. Lu, M. Mitchell, D. Batra, C. L. Zitnick, and D. Parikh, **“Vqa: Visual question answering,”** in _Proceedings of the IEEE international conference on computer vision_, 2015,
- Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh, **“Making the v in vqa matter: Elevating the role of image understanding in visual question answering,”** in _Proceedings of the IEEE conference on computer vision and pattern recognition_, 2017
- W. Wang, H. Bao, L. Dong, J. Bjorck, Z. Peng, Q. Liu, K. Aggarwal, O. K. Mohammed, S. Singhal, S. Som, and F. Wei, **“Image as a foreign language: BEiT pretraining for vision and vision-language tasks,”** in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, 2023.
- G. Hinton, O. Vinyals, and J. Dean, **“Distilling the knowledge in a neural network,”** 2015.
