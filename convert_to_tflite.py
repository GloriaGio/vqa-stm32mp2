import argparse
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2

from utils.config import load_config
from data.dataset import get_vqav2
from data.text_processing import Tokenizer

#
#
#


def get_args():
    parser = argparse.ArgumentParser(
        "Script for TFLite conversion and (per-tensor) quantization."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Folder containing the model to convert.",
    )
    return parser.parse_args()


#
#
#


def TFLconversion(config):

    # Load the trained model
    arch = config["model"]["model_architecture"]
    model_path = config["paths"]["saving_folder"] / f"trained_{arch}.keras"
    model = keras.models.load_model(model_path)

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

    df_train = get_vqav2(
        config["paths"]["dataset_path"],
        split="train2014",
        keep_10ans=False,
        verbose=True,
    )
    df_train500 = df_train[["question", "image_name"]].sample(500, random_state=123)
    questions = list(df_train500["question"])
    images = list(df_train500["image_name"])

    # Representative dataset generator for post-training quantization
    def representative_data_gen():
        for quest, im_name in zip(questions, images):
            tok_question = tokenizer.texts_to_sequences([quest])[0]
            tok_question = np.expand_dims(tok_question, axis=0)

            image_path = config["paths"]["dataset_path"] / "train2014" / im_name
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

            yield [tok_question.astype(np.float32), image.astype(np.float32)]

    # Define a concrete function to fix input shapes (batch size = 1)
    im_size = config["model"]["image_size"]
    num_channels = config["model"]["num_channels"]
    maxlen = config["model"]["max_length"]
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(1, maxlen), dtype=tf.float32),
            tf.TensorSpec(shape=(1, im_size, im_size, num_channels), dtype=tf.float32),
        ]
    )
    def model_fn(question, image):
        return model([question, image])

    concrete_func = model_fn.get_concrete_function()

    # Create the TFLite converter
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    # Enable optimizations (quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Set the representative dataset for calibration
    converter.representative_dataset = representative_data_gen

    # Force full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Force per-tensor quantization instead of per-channel
    converter._experimental_disable_per_channel = True

    # Convert the model
    tflite_model = converter.convert()

    # Save the quantized TFLite model
    TFLmodel_path = config["paths"]["saving_folder"] / f"trained_{arch}.tflite"
    with open(TFLmodel_path, "wb") as f:
        f.write(tflite_model)


#
#
#

if __name__ == "__main__":

    # Parse command-line argument to get the directory of the trained model
    args = get_args()
    saving_folder = Path("outputs") / args.model_dir

    # Load used configuration from JSON file
    used_config = load_config(saving_folder / "used_config.json")

    # Load paths and adds them to the used configuration
    config = load_config()
    used_config["paths"] = config["paths"]
    used_config["paths"]["saving_folder"] = saving_folder

    TFLconversion(used_config)
