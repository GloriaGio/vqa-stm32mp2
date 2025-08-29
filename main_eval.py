import argparse
from pathlib import Path
import json

import pandas as pd
import tensorflow as tf
from tensorflow import keras

from utils.config import load_config
from data.dataset import get_vqav2
from data.text_processing import Tokenizer
from data.custom_generators import Custom_Generator
from train.performance import get_model_ans, vqa_accuracy

#
#
#


def get_args():
    parser = argparse.ArgumentParser(
        "Script to evaluate a model on a specific dataset."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Folder containing the model to be evaluated.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train2014", "val2014"],
        help="Dataset split to evaluate the model on.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size to use during evaluation.",
    )
    return parser.parse_args()


#
#
#


def prepare_and_generate_answers(df, config):
    # Load tokenizer
    if config["model"]["min_frequency"] > 0:
        mf = config["model"]["min_frequency"]
        tokenizer_path = config["paths"]["output_path"] / f"word_index_mf{mf}.json"
    else:
        num_words = config["model"]["num_vocab_words"]
        tokenizer_path = config["paths"]["output_path"] / f"word_index{num_words}.json"
    with open(tokenizer_path, "r", encoding="utf-8") as file:
        word_index = json.load(file)
    tokenizer = Tokenizer(word_index=word_index, maxlen=config["model"]["max_length"])

    # Load the list of possible answers from the JSON file
    ct = "ct" if config["model"]["consider_teacher"] else ""
    num_classes = config["model"]["num_classes"]
    possible_ans_path = (
        config["paths"]["output_path"] / f"possible_answers_{ct}{num_classes}.json"
    )
    with open(possible_ans_path, "r", encoding="utf-8") as file:
        possible_ans = json.load(file)

    # Load custom data generators directly for evaluation
    data_loader = Custom_Generator(
        df,
        config["paths"]["dataset_path"],
        tokenizer,
        onehot_encoder=None,
        im_size=config["model"]["image_size"],
        num_channels=config["model"]["num_channels"],
        sample_weights=False,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )

    # Load the trained model
    arch = config["model"]["model_architecture"]
    model_path = config["paths"]["saving_folder"] / f"trained_{arch}.keras"
    model = keras.models.load_model(model_path)

    # Generates model answers
    model_ans = get_model_ans(model, data_loader, possible_ans)

    # Saving in the right format
    question_ids = list(df["question_id"])
    model_ans_s = [
        {"question_id": id, "answer": ans} for id, ans in zip(question_ids, model_ans)
    ]

    return model_ans_s


#
#
#


def main(config, split, save=True):

    # Load complete (unfiltered) pandas dataFrame for the considered split
    df = get_vqav2(
        config["paths"]["dataset_path"], split=split, keep_10ans=True, verbose=True
    )

    model_ans_path = config["paths"]["saving_folder"] / f"{split}_answers.json"
    # Gets the model answers from the file if exists, else generates and saves them
    if model_ans_path.is_file():
        with open(model_ans_path, "r", encoding="utf-8") as file:
            model_ans_s = json.load(file)
    else:
        model_ans_s = prepare_and_generate_answers(df, config)
        if save:
            with open(model_ans_path, "w") as file:
                json.dump(model_ans_s, file)

    df_model_ans = pd.DataFrame(model_ans_s)
    df = pd.merge(df, df_model_ans, how="outer", on="question_id")

    # Compute accuracy
    ten_ans = list(df["normalized_10answers"])
    model_ans = list(df["answer"])
    accuracy_all = vqa_accuracy(model_ans, ten_ans)
    performance = {"all": accuracy_all}

    # Compute accuracy per answer type
    for ans_type in ["yes/no", "number", "other"]:
        sub_df = df[df["answer_type"] == ans_type]
        ten_ans = list(sub_df["normalized_10answers"])
        model_ans = list(sub_df["answer"])
        performance[ans_type] = vqa_accuracy(model_ans, ten_ans)

    print(f"{split} accuracy: {accuracy_all*100:.2f}")

    with open(config["paths"]["saving_folder"] / f"{split}_accuracy.json", "w") as file:
        json.dump({split: performance}, file)


#
#
#

if __name__ == "__main__":

    # Parse command-line argument to get the directory of the trained model
    args = get_args()
    saving_folder = Path("outputs") / args.model_dir
    split = args.split

    # Load used configuration from JSON file
    used_config = load_config(saving_folder / "used_config.json")
    if args.batch_size is not None:
        used_config["training"]["batch_size"] = args.batch_size

    # Load paths and adds them to the used configuration
    config = load_config()
    used_config["paths"] = config["paths"]
    used_config["paths"]["saving_folder"] = saving_folder

    # Run the main evaluation pipeline and compute VQA v2 accuracy
    # get_vqa_accuracy(used_config, split)
    main(used_config, split)
