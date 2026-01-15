# Teacher Logits Generation

This folder contains the reference code used to generate teacher logits for **knowledge distillation (KD)** in this project, using a **BEiT-3** model fine-tuned on VQAv2.

The generated logits are later consumed by the student models during KD training.

---

## Requirements

Teacher logits generation relies on the **original BEiT-3 implementation** and its dependencies.

Please refer to the official repository for setup instructions and required packages:

https://github.com/microsoft/unilm/blob/master/beit3/README.md

The teacher code must be cloned locally and placed inside this folder (` teacher_logits_generation/`). **The provided script should then be moved into `unilm/beit3/`, as it relies on functions and utilities from the teacher repository.**

Additionally, the **BEiT-3 tokenizer** provided by the teacher repository is required.  
It can be downloaded [here](https://github.com/addf400/files/releases/download/beit3/beit3.spm) and should also be placed in this folder (`teacher_logits_generation/`).

---

## Teacher Model Weights

The teacher weights used in this project are:

- **Model:** `beit3_large_indomain_patch16_224 (480×480)`
- **Direct download:**  
  (https://github.com/addf400/files/releases/download/beit3/beit3_large_indomain_patch16_480_vqa.pth)

The downloaded weights should be placed in this folder (` teacher_logits_generation/`).

---

## Dataset

Logits are generated on the **VQAv2 dataset**.  
Please make sure the dataset is downloaded and structured as described in the main project README.

---

## Notes

- The provided code is intended as **reference** for reproducibility.
- It uses **PyTorch**, while the student models are implemented in **TensorFlow/Keras**.
