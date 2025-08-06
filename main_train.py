import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

from utils.config import load_config
from data.dataset import get_filtered_trainval
from data.text_processing import save_tokenizer
from data.onehot_encoder import OneHotEncoder
from data.custom_generators import get_custom_generators
from models.vqa_models import get_model
from train.train import KD_train_model, scratch_train_model


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


def main(config):

    if config["training"]["knowledge_distillation"]:
        config["model"]["consider_teacher"] = True
    else:
        config['training']['temperature'] = -1
        config['training']['alpha'] = -1

    df_train, df_val = get_filtered_trainval(
        config, consider_teacher=config["model"]["consider_teacher"], verbose=True
    )

    if config["model"]["num_vocab_words"] > 0:
        config["model"]["min_frequency"] = 0
        num_words = config["model"]["num_vocab_words"]
        tokenizer_path = config["paths"]["output_path"] / f"word_index{num_words}.json"
    else:
        mf = config["model"]["min_frequency"]
        tokenizer_path = config["paths"]["output_path"] / f"word_index_mf{mf}.json"
    if not tokenizer_path.is_file():
        save_tokenizer(config, tokenizer_path, verbose=False)

    ct = 'ct' if config["model"]["consider_teacher"] else ''
    num_classes = config["model"]["num_classes"]
    possible_ans_path = config["paths"]["output_path"] / f"possible_answers_{ct}{num_classes}.json"
    if not possible_ans_path.is_file():
        enc = OneHotEncoder()
        train_ans = list(df_train["normalized_answer"])
        enc.fit(train_ans)
        enc.save_json(possible_ans_path)

    train_data, valid_data = get_custom_generators(
        df_train, df_val, tokenizer_path, possible_ans_path, config
    )

    model = get_model(config, tokenizer_path)

    if config["training"]["knowledge_distillation"]:
        model = KD_train_model(model, train_data, valid_data, config)
        train_data, valid_data = get_custom_generators(
            df_train, df_val, tokenizer_path, possible_ans_path, config, get_logits=False
        )
        pass
    else:
        model = scratch_train_model(model, train_data, valid_data, config)
        pass

    arch = config["model"]["model_architecture"]
    model.save(config['paths']['saving_folder'] / f"trained_{arch}.keras")

    # calcolo accuracy normale ecc

    pass


if __name__ == "__main__":

    args = get_args()
    config = load_config()

    if args.model_arch is not None:
        config["model"]["model_architecture"] = args.model_arch
    if args.num_epochs is not None:
        config["training"]["num_epochs"] = args.num_epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    now = datetime.now().strftime("%y%m%d_%H%M")
    model_arch = config["model"]["model_architecture"]
    saving_folder = config["paths"]["output_path"] / f"{model_arch}_{now}"
    saving_folder.mkdir(parents=True, exist_ok=True)

    config["paths"]["saving_folder"] = saving_folder

    main(config)

    # dopo che è stata modificata e sono state aggiunte cose e tutto
    config.pop("paths", None)
    with open(saving_folder / "used_config.json", "w") as f:
        json.dump(config, f, indent=3)
        # ricordarmi di mettere a -1 temperature, alpha o num_glimps nella config (e che altro)
