# Visual Question Answering (VQA) on Edge Devices

This repository provides a hands-on tutorial, implemented in **Keras**, on how to design and train **Visual Question Answering (VQA)** models for **resource-constrained edge devices**, specifically the **STM32MP2 platform**.  
The workflow includes:

1. Model architecture design
2. Dataset preparation and preprocessing
3. Training with Knowledge Distillation (KD)
4. Model deployment on STM32MP2

**NOTE:**: 1. aggiungo tabella/e risultati 2. aggiungo risultati su STM32MP2 3. sistemo come usarlo (ad alto livello) 4. aggiungo dettagli su codice 5. scrivo qui in alto come è strutturato readme e dove trovo cosa

---

## 1. Model Architectures

The implemented models are based on the architectures proposed by Yu et al. (2017):

- **MFB Baseline**

  - Image features: extracted with ResNet-152 pre-trained on ImageNet.
  - Question features: word embeddings + two-layer LSTM network.
  - Fusion: MFB module combines image & question features.
  - Classifier selects the most likely answer from a predefined set of possible answers.

  ![MFB Module](Images/MFBModule.png)

  *Figure 1. MFB module proposed by Yu et a. (2017), *k* and *o\* are hyper-parameters of the module.

- **MFB + Attention**

  - Same as Baseline, with an **attention mechanism over image features**.
  - Uses two MFB modules in sequence.

- **MFB + CoAttention**
  - Most powerful variant.
  - Same as MFB + Attention, but applies **attention over both image and question features**.

### Optimizations for STM32MP2

The original architectures were modified to make them efficient on the STM32MP2 platform and leverage hardware acceleration (designed for forward CNNs):

- **ResNet-152 → MobileNet V3 Large**
  - Lighter, optimized for edge devices.
- **LSTM → Block of Temporal Convolutional Layers (TCLs)**
  - Faster and hardware-friendly sequence modeling.
- **MFB module simplification**
  - Removed power normalization
  - Sum pooling → average pooling
  - Parameters: `k = 5`, `o = 1024`

Result: efficient models suitable for deployment on edge devices.

![MFBCoAttention Architecture](Images/MFBCoAttention.png)

_Figure 2. Optimized MFB with Co-Attention network. The MFB with Attention and Baseline variants are derived by removing, respectively, the question attention block, and both the question and image attention blocks along with the first MFB module._

![TC Block](Images/TCLsBlock.png)

_Figure 3. Temporal Convolutional (TC) block replacing LSTM_

---

## 2. Dataset and Preprocessing

The tutorial uses the **VQAv2 dataset** (Goyal et al., 2017).  
Each sample contains:

- An image
- A natural language question
- Ten human-provided answers + one ground truth answer

Dataset: [VQAv2 official website](https://visualqa.org/)

### Dataset splits

| Set        | Questions | Images |
| ---------- | --------- | ------ |
| Training   | 443,757   | 82,783 |
| Validation | 214,354   | 40,504 |
| Test       | 447,793   | 81,434 |

Because the test set does not include ground truth or human-provided answers, all evaluations in this tutorial were carried out on the validation set.

### Answer preprocessing

- Normalized (lowercase, digits, removed articles, etc.).
- Only the **1000 most frequent answers** retained (standard VQA practice).
- Training set reduced to ~87.5% of original size, and this reduced set was used for training.
- As a result, the model can only predict one of these 1000 possible answers.

### Image preprocessing

- Resized to **224×224**.
- Normalized to range **[-1, 1]** (MobileNet input format).

### Question preprocessing

- Tokenized and padded/truncated to **15 tokens**.
- Vocabulary: **6415 words** (words appearing ≥5 times).
- Unknown words replaced with `<unk>`.

---

## 3. Training Procedure

### Knowledge Distillation (KD)

Instead of training from scratch, models are trained using **knowledge distillation**:

**Teacher:** BEiT-3 fine-tuned on VQAv2 [(beit3_large_indomain_patch16_224)](https://github.com/microsoft/unilm/tree/master/beit3#fine-tuning-on-vqav2-visual-question-answering) (82.53% accuracy on test-dev, 683M parameters).

### Loss function

The loss combines two terms:

- **Student loss**: cross-entropy (CE) with ground truth labels (_y_).
- **Distillation loss**: Kullback–Leibler (KL) divergence between teacher logits (_t_) & student logits (_s_).

\[
\mathcal{L} = \alpha \cdot \mathcal{L}_\text{student} + (1 - \alpha) \cdot \mathcal{L}_\text{distillation}
\]

\[
\mathcal{L}\_\text{student} = \mathrm{CE} \left(y, \, \mathrm{Softmax}(s) \right)
\]

\[
\mathcal{L}\_\text{distillation} = T^2 \cdot \mathrm{KL} \left( \mathrm{Softmax} (\frac{t}{T} ) \: || \: \mathrm{Softmax} (\frac{s}{T} ) \right)
\]

Parameters used:

- Temperature `T = 3`
- Balance coefficient `α = 0.1`

### Class imbalance handling

Since _yes/no_ answers represent ~40% of the dataset, **sample weights** inversely proportional to answer frequency were applied.

### Training setup

- GPU: NVIDIA GeForce GTX 1060 (6 GB)
- Optimizer: **Adam**, learning rate `1e-4`
- Epochs: **10** (~1 hour per epoch, ~10 hours total per model)

---

## 4. Deployment on STM32MP2

After training, models can be exported and optimized for inference:

- Convert trained models to a format supported by STM32MP2 runtime.
- Use hardware acceleration (optimized for CNN forward passes).
- Run VQA inference in real time.

**NOTA:** tabella tempi, energia ecc.

---

## 5. Usage

### Data Setup and External Resources

Before running the code, make sure to download and place the required external resources in the appropriate folders:

1. **VQAv2 Dataset**

- Download the dataset from the [official website](https://visualqa.org/).
- Place the images and annotation files in a folder like:
  ```
  data/vqa_dataset/
      ├── train2014/
      ├── val2014/
      ├── v2_mscoco_train2014_annotations.json
      ├── v2_mscoco_val2014_annotations.json
      ├── v2_OpenEnded_mscoco_train2014_questions.json
      └── v2_OpenEnded_mscoco_val2014_questions.json
  ```

2. **Teacher model logits (for Knowledge Distillation)**

   - Precomputed logits from the BEiT-3 teacher model (or any other teacher model) are required if you want to train using knowledge distillation (KD).
   - Save them in a folder structure like this:

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

   - **Optional:** If teacher logits are not available, you can still train the model from scratch by adjusting the configuration file accordingly.

3. **GloVe embeddings**

   - Download GloVe embeddings (i.e., `glove.6B.zip`) from [GloVe site](https://nlp.stanford.edu/projects/glove/).
   - Unzip the file and place it in:
     ```
     data/glove/
          └── glove.6B
     ```
   - **Optional:** you can still train the model without GloVe embeddings by adjusting the configuration file accordingly.

### Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/username/vqa-stm32mp2.git
cd vqa-stm32mp2
pip install -r requirements.txt
```

### Training a model

**nota**: o modifichi il file config, o dai da terminale

```bash
python train.py --model mfb_coatt --epochs 10 --batch_size 64
```

### Evaluation

**nota**: da terminale dai almeno la cartella

```bash
python eval.py --checkpoint checkpoints/mfb_coatt.pth
```

### Deployment

**NOTA**: codice che salva in tflite o simile

```bash
# Example (to be completed with STM32MP2 deployment steps)
python export.py --checkpoint checkpoints/mfb_coatt.pth --backend stm32mp2
```

---

## 6. Code Details

**Nota:** entro meglio nel codice.

---

## References

- Z. Yu, J. Yu, J. Fan, and D. Tao, "Multi-modal factorized bilinear pooling with co-attention learning for visual question answering," in _Proceedings of the IEEE international conference on computer vision_, 2017
- S. Antol, A. Agrawal, J. Lu, M. Mitchell, D. Batra, C. L. Zitnick, and D. Parikh, “Vqa: Visual question answering,” in _Proceedings of the IEEE international conference on computer vision_, 2015,
- Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh, “Making the v in vqa matter: Elevating the role of image understanding in visual question answering,” in _Proceedings of the IEEE conference on computer vision and pattern recognition_, 2017
- W. Wang, H. Bao, L. Dong, J. Bjorck, Z. Peng, Q. Liu, K. Aggarwal, O. K. Mohammed, S. Singhal, S. Som, and F. Wei, “Image as a foreign language: BEiT pretraining for vision and vision-language tasks,” in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, 2023.
- G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” 2015.
