import argparse
from pathlib import Path
import json

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
    parser = argparse.ArgumentParser("eval a VQA model")
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Model to eval folder",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size",
    )
    return parser.parse_args()


#
#
#


def get_vqa_accuracy(config):

    # Load complete (unfiltered) pandas dataFrames for training and validation sets
    df_train = get_vqav2(
        config["paths"]["dataset_path"],
        split="train2014",
        keep_10ans=True,
        verbose=True,
    )
    df_val = get_vqav2(
        config["paths"]["dataset_path"], split="val2014", keep_10ans=True, verbose=True
    )

    # Extract the 10 answers associated with each question from the dataframes
    train_10ans = list(df_train["normalized_10answers"])
    val_10ans = list(df_val["normalized_10answers"])

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
    train_data = Custom_Generator(
        df_train,
        config["paths"]["dataset_path"],
        tokenizer,
        onehot_encoder=None,
        im_size=config["model"]["image_size"],
        num_channels=config["model"]["num_channels"],
        sample_weights=False,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )
    valid_data = Custom_Generator(
        df_val,
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

    # Compute VQA v2 accuracy on train and validation sets and save performance metrics
    model_ans_train = get_model_ans(model, train_data, possible_ans)
    train_accuracy = vqa_accuracy(model_ans_train, train_10ans)

    model_ans_val = get_model_ans(model, valid_data, possible_ans)
    val_accuracy = vqa_accuracy(model_ans_val, val_10ans)

    print(
        f"Train Accuracy: {train_accuracy*100:.2f}, Val Accuracy: {val_accuracy*100:.2f}"
    )

    performance = {"train_accuracy": train_accuracy, "val_accuracy": val_accuracy}
    with open(config["paths"]["saving_folder"] / "performance.json", "w") as file:
        json.dump(performance, file)


#
#
#

if __name__ == "__main__":

    # # Parse command-line argument to get the directory of the trained model
    args = get_args()
    saving_folder = Path(args.folder)

    # Load used configuration from JSON file
    used_config = load_config(saving_folder / "used_config.json")
    if args.batch_size is not None:
        used_config["training"]["batch_size"] = args.batch_size

    # Load paths and adds them to the used configuration
    config = load_config()
    used_config["paths"] = config["paths"]
    used_config["paths"]["saving_folder"] = saving_folder

    # Run the main evaluation pipeline and compute VQA v2 accuracy
    get_vqa_accuracy(used_config)
