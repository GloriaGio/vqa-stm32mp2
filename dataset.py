import json
from pathlib import Path
import pandas as pd
from glossary import normalize_word
import config


def get_image_name(image_id, which_set="train2014"):
    image_id = str(image_id)
    while len(image_id) < 12:
        image_id = "0" + image_id
    image_id = "COCO_" + which_set + "_" + image_id + ".jpg"
    return image_id


def get_ans_list(dict_list):
    ans = []
    for dict_ans in dict_list:
        ans.append(normalize_word(dict_ans["answer"]))
    return ans


def get_vqav2(dataset_path, train=True, keep_10ans=False, verbose=True):

    if train:
        split = "train2014"
    else:
        split = "val2014"

    if verbose:
        print("Set: ", split)

    path_ans = dataset_path / f"v2_mscoco_{split}_annotations.json"
    with open(path_ans, "r", encoding="utf-8") as file:
        json_ans = json.load(file)

    df_ans = pd.DataFrame(json_ans["annotations"])
    if keep_10ans:
        df_ans["normalized_10answers"] = df_ans["answers"].apply(
            lambda x: get_ans_list(x)
        )
    df_ans = df_ans.drop(columns=["answers", "image_id"])

    path_q = dataset_path / f"v2_OpenEnded_mscoco_{split}_questions.json"
    with open(path_q, "r", encoding="utf-8") as file:
        json_q = json.load(file)

    df_q = pd.DataFrame(json_q["questions"])

    df = pd.merge(df_ans, df_q, how="outer", on="question_id")
    df["image_name"] = df["image_id"].apply(lambda x: get_image_name(x, split))
    df["normalized_answer"] = df["multiple_choice_answer"].apply(
        lambda x: normalize_word(x)
    )

    if verbose:
        print(f"Total number of sample in {split}: {len(df)}")

    return df


if __name__ == "__main__":
    df_train = get_vqav2(
        config.dataset_path, train=True, keep_10ans=False, verbose=True
    )
    print("Columns: ", [i for i in df_train.columns])
    df_val = get_vqav2(config.dataset_path, train=False, keep_10ans=False, verbose=True)
    print("Columns: ", [i for i in df_val.columns])
