import argparse
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2

from utils.config import load_config
from data.text_processing import Tokenizer

#
#
#


def get_args():
    parser = argparse.ArgumentParser(
        "Script for inference: given an image and a question, outputs an answer."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Folder containing the model to use for inference.",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to answer.",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to the image.",
    )
    return parser.parse_args()


#
#
#


def inference(question, image_path, config, verbose=True):
    # Load tokenizer
    if config["model"]["min_frequency"] > 0:
        mf = config["model"]["min_frequency"]
        tokenizer_path = Path("outputs") / f"word_index_mf{mf}.json"
    else:
        num_words = config["model"]["num_vocab_words"]
        tokenizer_path = Path("outputs") / f"word_index{num_words}.json"
    with open(tokenizer_path, "r", encoding="utf-8") as file:
        word_index = json.load(file)
    tokenizer = Tokenizer(word_index=word_index, maxlen=config["model"]["max_length"])

    # Load the list of possible answers from the JSON file
    ct = "ct" if config["model"]["consider_teacher"] else ""
    num_classes = config["model"]["num_classes"]
    possible_ans_path = Path("outputs") / f"possible_answers_{ct}{num_classes}.json"
    with open(possible_ans_path, "r", encoding="utf-8") as file:
        possible_ans = json.load(file)

    # Load the trained model
    arch = config["model"]["model_architecture"]
    model_path = config["paths"]["saving_folder"] / f"trained_{arch}.keras"
    model = keras.models.load_model(model_path)

    # Tokenized question
    tok_question = tokenizer.texts_to_sequences([question])

    # Processed image
    if config["model"]["num_channels"] == 1:
        how = cv2.IMREAD_GRAYSCALE
        how2 = cv2.COLOR_GRAY2RGB
    else:
        how = cv2.IMREAD_COLOR
        how2 = cv2.COLOR_BGR2RGB
    image = cv2.imread(image_path, how)
    image = cv2.cvtColor(image, how2)
    size = config["model"]["image_size"]
    image = cv2.resize(image, (size, size))
    image = image.astype(dtype="float32") / 255.0 * 2 - 1
    image = np.expand_dims(image, axis=0)

    # Model answer
    pred = model.predict((tok_question, image), verbose=0)
    ans_idx = pred[0].argmax()
    answer = possible_ans[ans_idx]

    if verbose:
        print(f"Question: {question}")
        print(f"Model answer: {answer}")

    return answer


#
#
#

if __name__ == "__main__":

    # Parse command-line argument to get the directory of the trained model
    args = get_args()
    question = args.question
    image_path = Path(args.image_path)
    saving_folder = Path("outputs") / args.model_dir

    # Load used configuration from JSON file
    used_config = load_config(saving_folder / "used_config.json")
    used_config["paths"]["saving_folder"] = saving_folder

    ans = inference(question, image_path, used_config, verbose=True)
