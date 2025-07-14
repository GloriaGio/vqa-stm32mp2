import config
from dataset import get_vqav2
from text_processing import Tokenizer
from custom_generators import Custom_Generator

from tensorflow import keras
import numpy as np
import argparse
from pathlib import Path
import json


def vqa_accuracy(model_ans, list_10ans):
    count = 0
    for i in range(len(model_ans)):
        count += min(list_10ans[i].count(model_ans[i]) / 3.0, 1.0)
    return count / len(model_ans)

def get_args():
    parser = argparse.ArgumentParser("VQA model: computing accuracy")
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Model folder",
    )
    parser.add_argument(
        "--batch_size",
        default=10000,
        type=int,
        help="Batch size",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    BS_SIZE = min(args.batch_size, config.BS_SIZE)
    saving_folder = config.folder_path / args.folder

    dataset_path = config.dataset_path

    ### DATA LOADING ###

    df_train = get_vqav2(dataset_path, train=True, keep_10ans=True, verbose=True)
    df_train = df_train[:5000]
    df_val = get_vqav2(dataset_path, train=False, keep_10ans=True, verbose=True)
    df_val = df_val[:5000]

    # Tokenizer
    with open(config.folder_path/"tokenizer_word_index.json", "r", encoding="utf-8") as file:
        word_index = json.load(file)
    tokenizer = Tokenizer(word_index=word_index, maxlen=config.maxlen)
    num_words = len(word_index)
    
    # Train data loader
    train_data = Custom_Generator(
        df_train, 
        config.dataset_path,
        tokenizer,
        onehot_encoder=None,
        im_size=config.im_size,
        num_channels=config.num_channels,
        sample_weights=False,
        batch_size=BS_SIZE,
        shuffle=False,
    )

    # Vaild data loader
    valid_data = Custom_Generator(
        df_val,
        config.dataset_path,
        tokenizer,
        onehot_encoder=None,
        im_size=config.im_size,
        num_channels=config.num_channels,
        sample_weights=False,
        batch_size=BS_SIZE,
        shuffle=False,
    )

    # possible answers
    with open(saving_folder/"possible_answers.json", "r", encoding="utf-8") as file:
        possible_ans = json.load(file)

    # models
    final_model = keras.models.load_model(saving_folder / "final_model.keras")
    try:
        best_model = keras.models.load_model(saving_folder / "best_model.keras")
    except:
        best_model = None

    final_model_perf = {}

    # final model on train set
    train_pred = final_model.predict(train_data)
    train_ans_idx = train_pred.argmax(axis=-1)
    train_ans = [possible_ans[idx] for idx in train_ans_idx]

    train_10ans = list(df_train["normalized_10answers"])
    final_model_perf['train_accuracy'] =  vqa_accuracy(train_ans, train_10ans)

    # final model on val set
    val_pred = final_model.predict(valid_data)
    val_ans_idx = val_pred.argmax(axis=-1)
    val_ans = [possible_ans[idx] for idx in val_ans_idx]

    val_10ans = list(df_val["normalized_10answers"])
    final_model_perf['val_accuracy'] =  vqa_accuracy(val_ans, val_10ans)

    print("Final model:")
    print(f"- Train Accuracy: {final_model_perf['train_accuracy']*100:.2f} %, Val Accuracy: {final_model_perf['val_accuracy']*100:.2f} %")

    if best_model is not None:
        best_model_perf = {}

        # best model on train set
        train_pred = best_model.predict(train_data)
        train_ans_idx = train_pred.argmax(axis=-1)
        train_ans = [possible_ans[idx] for idx in train_ans_idx]

        best_model_perf['train_accuracy'] =  vqa_accuracy(train_ans, train_10ans)

        # bets model on val set
        val_pred = best_model.predict(valid_data)
        val_ans_idx = val_pred.argmax(axis=-1)
        val_ans = [possible_ans[idx] for idx in val_ans_idx]

        best_model_perf['val_accuracy'] =  vqa_accuracy(val_ans, val_10ans)

        print("Best model:")
        print(f"- Train Accuracy: {best_model_perf['train_accuracy']*100:.2f} %, Val Accuracy: {best_model_perf['val_accuracy']*100:.2f} %")

performance = {
    'final_model': final_model_perf,
    'best_model': best_model_perf
}
with open(saving_folder / "accuracy.json", "w") as file:
    json.dump(performance, file)
