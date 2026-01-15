import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import pandas as pd
from transformers import XLMRobertaTokenizer
from torchvision.datasets.folder import default_loader

import modeling_finetune
from datasets import build_transform
from glossary import normalize_word

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Paths
# -----------------------------

weights_file = "beit3_large_indomain_patch16_480_vqa.pth"

repo_path = Path(__file__).resolve().parents[3]
teacher_logits_path = repo_path / "teacher_logits_generation"
resources_path = repo_path / "resources"

dataset_path = repo_path / "vqa_dataset"

weights_path = teacher_logits_path / weights_file
tokenizer_path = teacher_logits_path / "beit3.spm"
answer2label_path = resources_path / "teacher_logits" / "answer2label.txt"

train_logits_folder = resources_path / "teacher_logits" / "train_logits"
val_logits_folder = resources_path / "teacher_logits" / "val_logits"
train_logits_folder.mkdir(parents=True, exist_ok=True)
val_logits_folder.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Model and Tokenizer
# -----------------------------
tokenizer = XLMRobertaTokenizer(tokenizer_path)
model = modeling_finetune.beit3_large_patch16_480_vqav2()

data = torch.load(weights_path)
weights = data["model"]
model.load_state_dict(weights)
model.eval()
model.to(device)

# -----------------------------
# Load Answers
# -----------------------------
labels_answers = {}
answers = []
with open(answer2label_path, encoding="utf-8") as file:
    for row in file:
        diz = json.loads(row.strip())
        labels_answers[diz["label"]] = diz["answer"]
        answers.append(diz["answer"])

# -----------------------------
# Image Transform
# -----------------------------
args = SimpleNamespace(
    input_size=480, task="vqav2", randaug=False, train_interpolation=3
)
transform = build_transform(False, args)


# -----------------------------
# Helper Function
# -----------------------------
def get_name_image(image_id, which_set="train2014"):
    image_id = str(image_id).zfill(12)
    return f"COCO_{which_set}_{image_id}.jpg"


# -----------------------------
# Load Dataset and Merge
# -----------------------------
def prepare_dataframe(ans_file, q_file, split="train2014"):
    with open(ans_file, "r", encoding="utf-8") as f:
        ans = json.load(f)
    with open(q_file, "r", encoding="utf-8") as f:
        q = json.load(f)
    df_ans = pd.DataFrame(ans["annotations"]).drop(columns="answers")
    df_q = pd.DataFrame(q["questions"]).drop(columns="image_id")
    df = pd.merge(df_ans, df_q, how="outer", on="question_id")
    return df


df_train = prepare_dataframe(
    dataset_path / "v2_mscoco_train2014_annotations.json",
    dataset_path / "v2_OpenEnded_mscoco_train2014_questions.json",
    "train2014",
)
df_val = prepare_dataframe(
    dataset_path / "v2_mscoco_val2014_annotations.json",
    dataset_path / "v2_OpenEnded_mscoco_val2014_questions.json",
    "val2014",
)


# -----------------------------
# Function to generate logits
# -----------------------------
def generate_logits(df, split_folder, dataset_split):
    r = len(df)
    count = 0

    for i in range(r):
        row = dict(df.iloc[i])
        row["image_id"] = row["image_id"].item()
        row["question_id"] = row["question_id"].item()

        im_name = row["image_name"]
        quest = row["question"]
        gt = row["multiple_choice_answer"]

        # Tokenize question
        encoded = tokenizer(
            [quest], padding=True, truncation=False, return_tensors="pt"
        )
        tokenized_quest = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        padding_mask = attention_mask == 0

        # Load and transform image
        im_path = dataset_path / dataset_split / im_name
        im = default_loader(im_path)
        im = transform(im)
        im = torch.unsqueeze(im, 0)

        # Model prediction
        with torch.no_grad():
            pred = model(
                im.to(device), tokenized_quest.to(device), padding_mask.to(device)
            )
            row["logits"] = pred[0].tolist()
            pred_index = int(torch.argmax(pred))
            row["teacher_answer"] = labels_answers[pred_index]

        # Normalize answer and save
        row["normalized_answer"] = normalize_word(row["multiple_choice_answer"])
        file_name = f"{row['question_id']}.json"
        with open(split_folder / file_name, "w") as f:
            json.dump(row, f)

        # Accuracy count (optional)
        if labels_answers[pred_index] == gt:
            count += 1

    print(f"Processed {r} samples from {dataset_split}, accuracy: {count/r:.4f}")


# -----------------------------
# Generate logits for train and val
# -----------------------------
generate_logits(df_train, train_logits_folder, "train2014")
generate_logits(df_val, val_logits_folder, "val2014")
