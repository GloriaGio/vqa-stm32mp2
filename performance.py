import config
from dataset import get_vqav2
from text_processing import Tokenizer
from custom_generators import Custom_Generator

from tensorflow import keras
import numpy as np
import argparse
from pathlib import Path
import json


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


def vqa_accuracy(model_ans, list_10ans):
    count = 0
    for i in range(len(model_ans)):
        count += min(list_10ans[i].count(model_ans[i]) / 3.0, 1.0)
    return count / len(model_ans)


def get_vqa_accuracy(model, dataloader, possible_ans, gt_10ans):
    model_ans = []
    for i in range(len(dataloader)):
        batch_input, _, _ = dataloader[i]
        pred = model.predict(batch_input, verbose=0)
        ans_idx = pred.argmax(axis=-1)
        batch_ans = [possible_ans[idx] for idx in ans_idx]
        model_ans += batch_ans

    accuracy = vqa_accuracy(model_ans, gt_10ans)
    return accuracy




if __name__ == "__main__":

    args = get_args()
    BS_SIZE = min(args.batch_size, config.BS_SIZE)
    saving_folder = config.trained_models_path / args.folder

    dataset_path = config.dataset_path

    ### DATA LOADING ###

    df_train = get_vqav2(dataset_path, train=True, keep_10ans=True, verbose=True)
    train_10ans = list(df_train["normalized_10answers"])

    df_val = get_vqav2(dataset_path, train=False, keep_10ans=True, verbose=True)
    val_10ans = list(df_val["normalized_10answers"])

    # Tokenizer
    with open(config.trained_models_path/"tokenizer_word_index.json", "r", encoding="utf-8") as file:
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

    performance = {}

    # final model, all answers
    final_model_perf = {}
    print("Final model:")
    final_model_perf['train_accuracy'] = get_vqa_accuracy(final_model, train_data, possible_ans, train_10ans)
    print(f"Train Accuracy: {final_model_perf['train_accuracy']*100:.2f} %")
    final_model_perf['val_accuracy'] = get_vqa_accuracy(final_model, valid_data, possible_ans, val_10ans)
    print(f"Val Accuracy: {final_model_perf['val_accuracy']*100:.2f} %")

    performance['final_model'] = final_model_perf

    # best model, all answers
    if best_model is not None:
        best_model_perf = {}
        print("Best model:")
        best_model_perf['train_accuracy'] = get_vqa_accuracy(best_model, train_data, possible_ans, train_10ans)
        print(f"Train Accuracy: {best_model_perf['train_accuracy']*100:.2f} %")
        best_model_perf['val_accuracy'] = get_vqa_accuracy(best_model, valid_data, possible_ans, val_10ans)
        print(f"Val Accuracy: {best_model_perf['val_accuracy']*100:.2f} %")

        performance['best_model'] = best_model_perf
    

    with open(saving_folder / "accuracy.json", "w") as file:
        json.dump(performance, file)