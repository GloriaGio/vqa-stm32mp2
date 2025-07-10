import json
from pathlib import Path
import pandas as pd
from glossary import normalize_word


def get_image_name(image_id, which_set="train2014"):
    image_id = str(image_id)
    while len(image_id) < 12:
        image_id = "0" + image_id
    image_id = "COCO_" + which_set + "_" + image_id + ".jpg"
    return image_id


def get_vqav2_training(dataset_path, verbose=True):
    if verbose:
        print("- Training set:")
    path_ans = dataset_path / "v2_mscoco_train2014_annotations.json"
    with open(path_ans, "r", encoding="utf-8") as file:
        train_ans = json.load(file)

    df_ans = pd.DataFrame(train_ans["annotations"])
    df_ans = df_ans.drop(columns=["answers", "image_id"])

    path_q = dataset_path / "v2_OpenEnded_mscoco_train2014_questions.json"
    with open(path_q, "r", encoding="utf-8") as file:
        train_q = json.load(file)

    df_q = pd.DataFrame(train_q["questions"])

    df_train = pd.merge(df_ans, df_q, how="outer", on="question_id")
    df_train["image_name"] = df_train["image_id"].apply(lambda x: get_image_name(x))
    df_train['normalized_answer'] = df_train['multiple_choice_answer'].apply(lambda x: normalize_word(x))

    if verbose:
        print(f"Total number of sample: {len(df_train)}")

    return df_train


def get_vqav2_validation(dataset_path, verbose=True):
    if verbose:
        print("- Validation set:")

    path_ans = dataset_path / "v2_mscoco_val2014_annotations.json"
    with open(path_ans, "r", encoding="utf-8") as file:
        val_ans = json.load(file)

    df_ans = pd.DataFrame(val_ans["annotations"])
    df_ans = df_ans.drop(columns=["answers", "image_id"])

    path_q = dataset_path / "v2_OpenEnded_mscoco_val2014_questions.json"
    with open(path_q, "r", encoding="utf-8") as file:
        val_q = json.load(file)

    df_q = pd.DataFrame(val_q["questions"])

    df_val = pd.merge(df_ans, df_q, how="outer", on="question_id")

    df_val["image_name"] = df_val["image_id"].apply(
        lambda x: get_image_name(x, "val2014")
    )
    df_val['normalized_answer'] = df_val['multiple_choice_answer'].apply(lambda x: normalize_word(x))

    if verbose:
        print(f"Total number of samples: {len(df_val)}")

    return df_val




if __name__ == "__main__":
    dataset_path = (
        Path.home() / "Desktop" / "VQA" / "vqa_dataset"
    )
    df_train = get_vqav2_training(dataset_path, True)
    print("Columns: ", [i for i in df_train.columns])
    df_val = get_vqav2_validation(dataset_path, True)
    print("Columns: ", [i for i in df_val.columns])
