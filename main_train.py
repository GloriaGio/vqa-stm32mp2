import argparse
from datetime import datetime
import json
from pathlib import Path

from utils.config import load_config
from data.dataset import get_filtered_trainval
from data.text_processing import save_tokenizer
from data.onehot_encoder import OneHotEncoder
from data.custom_generators import get_custom_generators
from models.vqa_models import get_model
from train.trainer import train_with_KD, train_from_scratch
from train.performance import preliminary_performance

#
#
#


def get_args():
    parser = argparse.ArgumentParser("train a VQA model")
    parser.add_argument(
        "--model-arch",
        type=str,
        choices=["MFBBaseline", "MFBAttention", "MFBCoAttention"],
        help="Model architecture to use: MFBBaseline, MFBAttention, MFBCoAttention",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="Maximum number of epochs",
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


def main(config):

    if config["training"]["knowledge_distillation"]:
        config["model"]["consider_teacher"] = True
    else:
        config["training"]["temperature"] = -1
        config["training"]["alpha"] = -1

    # Load pandas dataframes for training and validation,
    # filtered to include only the most frequent answers as specified in the config
    df_train, df_val = get_filtered_trainval(
        config, consider_teacher=config["model"]["consider_teacher"], verbose=True
    )

    # Check if tokenizer and one-hot encoder for the current config exist; create them if not
    if config["model"]["num_vocab_words"] > 0:
        config["model"]["min_frequency"] = 0
        num_words = config["model"]["num_vocab_words"]
        tokenizer_path = config["paths"]["output_path"] / f"word_index{num_words}.json"
    else:
        mf = config["model"]["min_frequency"]
        tokenizer_path = config["paths"]["output_path"] / f"word_index_mf{mf}.json"
    if not tokenizer_path.is_file():
        save_tokenizer(config, tokenizer_path, verbose=False)

    ct = "ct" if config["model"]["consider_teacher"] else ""
    num_classes = config["model"]["num_classes"]
    possible_ans_path = (
        config["paths"]["output_path"] / f"possible_answers_{ct}{num_classes}.json"
    )
    if not possible_ans_path.is_file():
        enc = OneHotEncoder()
        train_ans = list(df_train["normalized_answer"])
        enc.fit(train_ans)
        enc.save_json(possible_ans_path)

    # Prepare custom data generators for training and validation based on the config
    train_data, valid_data = get_custom_generators(
        df_train, df_val, tokenizer_path, possible_ans_path, config
    )

    # Initialize the model architecture as specified in the config
    model = get_model(config, tokenizer_path)

    # Train the model using knowledge distillation if enabled; otherwise, train from scratch
    if config["training"]["knowledge_distillation"]:
        model = train_with_KD(model, train_data, valid_data, config)
    else:
        model = train_from_scratch(model, train_data, valid_data, config)

    # Save the trained model
    arch = config["model"]["model_architecture"]
    model.save(config["paths"]["saving_folder"] / f"trained_{arch}.keras")

    # Evaluate preliminary accuracy and loss on filtered train and validation sets using standard accuracy (not VQA v2 metric)
    train_data, valid_data = get_custom_generators(
        df_train, df_val, tokenizer_path, possible_ans_path, config, get_logits=False
    )
    performance = preliminary_performance(model, train_data, valid_data, verbose=True)
    with open(
        config["paths"]["saving_folder"] / "preliminary_performance.json", "w"
    ) as file:
        json.dump(performance, file)


#
#
#

if __name__ == "__main__":

    # Parse command-line arguments to override default config values
    args = get_args()
    # Load model configuration from JSON file
    config = load_config()

    # Override default config values
    if args.model_arch is not None:
        config["model"]["model_architecture"] = args.model_arch
    if args.num_epochs is not None:
        config["training"]["num_epochs"] = args.num_epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    # Create directory to store the trained model, config, and performance metrics
    now = datetime.now().strftime("%y%m%d_%H%M")
    model_arch = config["model"]["model_architecture"]
    saving_folder = config["paths"]["output_path"] / f"{model_arch}_{now}"
    saving_folder.mkdir(parents=True, exist_ok=True)

    config["paths"]["saving_folder"] = saving_folder

    # Run the main training pipeline with the given config
    main(config)

    # Save the final configuration used for training
    config.pop("paths", None)
    with open(saving_folder / "used_config.json", "w") as f:
        json.dump(config, f, indent=3)
