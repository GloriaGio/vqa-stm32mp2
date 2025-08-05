import json
from collections import Counter

import pandas as pd

from data.glossary import normalize_word


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


def get_vqav2(dataset_path, split="train2014", keep_10ans=False, verbose=True):

    if verbose:
        print("Set: ", split)

    path_q = dataset_path / f"v2_OpenEnded_mscoco_{split}_questions.json"
    with open(path_q, "r", encoding="utf-8") as file:
        json_q = json.load(file)

    df_q = pd.DataFrame(json_q["questions"])
    if "dev" not in split:
        folder = split
    else:
        folder = "test2015"
    df_q["image_name"] = df_q["image_id"].apply(lambda x: get_image_name(x, folder))

    if verbose:
        print(f"Total number of sample in {split}: {len(df_q)}")

    if "test" in split:
        return df_q

    path_ans = dataset_path / f"v2_mscoco_{split}_annotations.json"
    with open(path_ans, "r", encoding="utf-8") as file:
        json_ans = json.load(file)

    df_ans = pd.DataFrame(json_ans["annotations"])
    df_ans["normalized_answer"] = df_ans["multiple_choice_answer"].apply(
        lambda x: normalize_word(x)
    )
    if keep_10ans:
        df_ans["normalized_10answers"] = df_ans["answers"].apply(
            lambda x: get_ans_list(x)
        )
    df_ans = df_ans.drop(columns=["answers", "image_id"])

    df = pd.merge(df_ans, df_q, how="outer", on="question_id")

    return df


def get_filtered_trainval(config, consider_teacher=True, verbose=True):

    # Training set
    df_train = get_vqav2(
        config["paths"]["dataset_path"],
        split="train2014",
        keep_10ans=False,
        verbose=verbose,
    )

    if consider_teacher:
        # teacher answers
        answers_labels = {}
        with open(
            config["paths"]["KD_path"] / "answer2label.txt", encoding="utf-8"
        ) as file:
            for row in file:
                diz = json.loads(row.strip())
                answers_labels[diz["answer"]] = diz["label"]
        teach_ans = list(answers_labels.keys())
    else:
        teach_ans = list(df_train["normalized_answer"])

    # most frequent normalized answers (that are also teacher answers if consider_teacher=True)
    freq_ans = Counter(df_train["normalized_answer"]).most_common()
    possible_ans = []
    weight_dict = {}
    for ans, freq in freq_ans:
        if ans in teach_ans:
            possible_ans.append(ans)
            weight_dict[ans] = 1 / freq

        if len(possible_ans) == config["model"]["num_classes"]:
            break
    config["model"]["num_classes"] = len(possible_ans)

    # filtered dataset (most frequent answers only)
    df_train_filtered = df_train[
        df_train["normalized_answer"].isin(possible_ans)
    ].copy()
    df_train_filtered["weight"] = df_train_filtered["normalized_answer"].apply(
        lambda x: weight_dict[x]
    )

    if verbose:
        print(
            f"Number of training samples after filtering: {len(df_train_filtered)} ({len(df_train_filtered)/len(df_train)*100: .2f} % )"
        )

    # Validation set
    df_val = get_vqav2(
        config["paths"]["dataset_path"],
        split="val2014",
        keep_10ans=False,
        verbose=verbose,
    )

    # filtered dataset
    df_val_filtered = df_val[df_val["normalized_answer"].isin(possible_ans)].copy()
    df_val_filtered["weight"] = df_val_filtered["normalized_answer"].apply(
        lambda x: weight_dict[x]
    )

    if verbose:
        print(
            f"Number of validation samples after filtering: {len(df_val_filtered)} ({len(df_val_filtered)/len(df_val)*100: .2f} % )"
        )

    return df_train_filtered, df_val_filtered


if __name__ == "__main__":

    print("??")
