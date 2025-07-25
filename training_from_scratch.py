import os

# hide INFO and WARNING
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from dataset import get_vqav2_training
from dataset import get_vqav2_validation
from vqa_models import get_net
from custom_generators import Custom_Generator

import argparse
import pandas as pd
import json
from pathlib import Path
import cv2
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random
from datetime import datetime
import pickle

import tensorflow as tf
from tensorflow import keras
import numpy as np

#
#
#
#
#


def get_args():
    parser = argparse.ArgumentParser("VQA model: training from scratch")
    parser.add_argument(
        "--net",
        type=str,
        required=True,
        choices=["tiny", "small", "big", "tinyCSA", "tinyCSA2"],
        help="Which net to use",
    )
    parser.add_argument(
        "--num_epochs",
        default=2,
        type=int,
        help="Maximum number of epochs",
    )
    return parser.parse_args()


#######################################################
print("NON SISTEMATO PER CSA A PIU BLOCCHI (ed eventuali altre mofiche!!")
#######################################################


args = get_args()
which_net = args.net


# project folder
path = Path.home() / "Desktop" / "VQA" / "VQAforMCUs"
# dataset folder
dataset_path = Path.home() / "Desktop" / "VQA" / "vqa_dataset"
# saving folder
now = datetime.now().strftime("%m_%d_%H_%M")
saving_folder = path / f"{which_net}_VQAmodel_{now}"
os.makedirs(saving_folder, exist_ok=True)

#
#
#
#
#

# input and output parameters
maxlen = 15  # maximum question length
num_words = 2000  # vocabulary number of words
im_size = 224  # image higth and width
num_channels = 1  # image channels
num_classes = 1000  # number of possible answers

# training parameters
NUM_EPOCHS = args.num_epochs  # epochs number
LR = 0.0001  # learning rate
BS_SIZE = 32  # batch size
DROPOUT_RATE = 0.1

#
#
#
#
#


### DATA LOADING ###

print("Loading the data...")

excluded_answ = get_excluded_answers()


# Training set

df_train = get_vqav2_training(dataset_path, False)

# most frequent answers (without excluded ones)
freq_ans = Counter(df_train["multiple_choice_answer"]).most_common()
most_freq = []
freqs = []
maxl = num_classes + len(excluded_answ)
for word, freq in freq_ans[:maxl]:
    if word not in excluded_answ:
        most_freq.append(word)
        freqs.append(freq)


# filtered dataset (most frequent answers only)
df_train_filtered = df_train[df_train["multiple_choice_answer"].isin(most_freq)].copy()


print(
    f"Number of training samples after filtering: {len(df_train_filtered)} ({len(df_train_filtered)/len(df_train)*100: .2f} % )"
)


list_train_q = list(df_train_filtered["question"])
list_train_gt = list(df_train_filtered["multiple_choice_answer"])
list_train_im = list(df_train_filtered["image_name"])

# Tokenization and padding
tokenizer = keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token="<OOV>")
tokenizer.fit_on_texts(list_train_q)
train_q = tokenizer.texts_to_sequences(list_train_q)
train_q = keras.preprocessing.sequence.pad_sequences(train_q, maxlen=maxlen)

# Onehot encoding
train_gt = np.reshape(np.array(list_train_gt), (-1, 1))
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
onehot_encoder.fit(train_gt)
train_gt = onehot_encoder.transform(train_gt)

# loss weights
weights = 1 / train_gt.sum(axis=0)

# data loader
train_data = Custom_Generator(
    dataset_path,
    train_q,
    list_train_im,
    train_gt,
    BS_SIZE,
    weights,
    shuffle=True,
    im_size=im_size,
    num_channels=num_channels,
)

#
#
#

# Validation set

df_val = get_vqav2_validation(dataset_path, False)

# filtered dataset
df_val_filtered = df_val[df_val["multiple_choice_answer"].isin(most_freq)].copy()

print(
    f"Number of validation samples after filtering: {len(df_val_filtered)} ({len(df_val_filtered)/len(df_val)*100: .2f} % )"
)


list_valid_q = list(df_val_filtered["question"])
list_valid_gt = list(df_val_filtered["multiple_choice_answer"])
list_valid_im = list(df_val_filtered["image_name"])

# Tokenization and padding
valid_q = tokenizer.texts_to_sequences(list_valid_q)
valid_q = keras.preprocessing.sequence.pad_sequences(valid_q, maxlen=maxlen)

# One hot encoding
valid_gt = np.reshape(np.array(list_valid_gt), (-1, 1))
valid_gt = onehot_encoder.transform(valid_gt)

valid_data = Custom_Generator(
    dataset_path,
    valid_q,
    list_valid_im,
    valid_gt,
    BS_SIZE,
    weights,
    shuffle=False,
    im_size=im_size,
    num_channels=num_channels,
)


#
#
#
#
#

### MODEL ###
model = get_net(
    which_net=which_net,
    maxlen=maxlen,
    num_words=num_words,
    im_size=im_size,
    num_channels=num_channels,
    num_classes=num_classes,
    dropout_rate=DROPOUT_RATE,
    last_softmax=False,
)

# model.summary()

#
#
#
#
#

### TRAINING ###

model.compile(
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=False
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=saving_folder / "best_model.h5",
        save_best_only=True,
        monitor="val_loss",
        initial_value_threshold=1.1,
        verbose=0,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        # cooldown=2,
        min_lr=5e-6,
        verbose=0,
    ),
]


history = model.fit(
    train_data, validation_data=valid_data, epochs=NUM_EPOCHS, callbacks=callbacks
)

#
#
#
#
#

### SAVING ###

# model saving
model.save(saving_folder / "final_model.h5")

# saving tokenizer
with open(saving_folder / "tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# saving onehot encoder
with open(saving_folder / "onehot_encoder.pickle", "wb") as handle:
    pickle.dump(onehot_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

# saving loss and accuracy trend
train_history = history.history
with open(saving_folder / "training_history.json", "w") as f:
    json.dump(train_history, f)

#
#
#
#
#

### PERFORMANCE ###

if True:

    final_model = keras.models.load_model(saving_folder / "final_model.h5")
    try:
        best_model = keras.models.load_model(saving_folder / "best_model.h5")
    except:
        best_model = None

    final_train_loss, final_train_accuracy = final_model.evaluate(train_data)
    final_val_loss, final_val_accuracy = final_model.evaluate(valid_data)

    performance = {}
    performance["final_model"] = {
        "train_loss": final_train_loss,
        "train_accuracy": final_train_accuracy,
        "val_loss": final_val_loss,
        "val_accuracy": final_val_accuracy,
    }

    print("Final model:")
    print(
        f"- Train Accuracy: {final_train_accuracy*100:.2f} %, Val Accuracy: {final_val_accuracy*100:.2f} %"
    )

    if best_model is not None:
        best_train_loss, best_train_accuracy = best_model.evaluate(train_data)
        best_val_loss, best_val_accuracy = best_model.evaluate(valid_data)
        performance["best_model"] = {
            "train_loss": best_train_loss,
            "train_accuracy": best_train_accuracy,
            "val_loss": best_val_loss,
            "val_accuracy": best_val_accuracy,
        }
        print("Best model:")
        print(
            f"- Train Accuracy: {best_train_accuracy*100:.2f} %, Val Accuracy: {best_val_accuracy*100:.2f} %"
        )

    with open(saving_folder / "performance.json", "w") as file:
        json.dump(performance, file)
