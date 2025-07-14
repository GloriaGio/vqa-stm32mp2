import os

# hide INFO and WARNING
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import config
from dataset import get_vqav2
from text_processing import Tokenizer, get_GloVe_emb
from custom_generators import Custom_Generator
from distiller import Distiller
from vqa_models import MFB_Baseline

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

def get_args():
    parser = argparse.ArgumentParser("VQA model: training with knowledge distillation")
    parser.add_argument(
        "--net",
        default='PROVA',
        type=str,
        #required=True,
        #choices=["tiny", "small", "big", "tinyCSA", "tinyCSA2"],
        help="Which net to use",
    )
    parser.add_argument(
        "--num_epochs",
        default=10000,
        type=int,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        default=10000,
        type=int,
        help="Batch size",
    )
    return parser.parse_args()


args = get_args()
which_net = args.net

# training parameters
NUM_EPOCHS = min(args.num_epochs, config.NUM_EPOCHS)
LR = config.LR
BS_SIZE = min(args.batch_size, config.BS_SIZE)

# saving folder
now = datetime.now().strftime("%y%m%d_%H%M")
saving_folder = config.folder_path / f"KD_{which_net}_{now}"
os.makedirs(saving_folder, exist_ok=True)


#
#
#

### DATA LOADING ###

print("Loading the data...")

# Training set
df_train = get_vqav2(config.dataset_path, train=True, keep_10ans=False, verbose=True)

# most frequent normalized answers 
freq_ans = Counter(df_train["normalized_answer"]).most_common(config.num_classes)
possible_ans = [ans for ans, _ in freq_ans]
weight_dict = {ans: 1/freq for ans, freq in freq_ans}

# filtered dataset (most frequent answers only)
df_train_filtered = df_train[df_train["normalized_answer"].isin(possible_ans)].copy()
df_train_filtered['weight'] = df_train_filtered["normalized_answer"].apply(lambda x: weight_dict[x])

print(
    f"Number of training samples after filtering: {len(df_train_filtered)} ({len(df_train_filtered)/len(df_train)*100: .2f} % )"
)

# Validation set
df_val = get_vqav2(config.dataset_path, train=False, keep_10ans=False, verbose=True)

# filtered dataset
df_val_filtered = df_val[df_val["normalized_answer"].isin(possible_ans)].copy()
df_val_filtered['weight'] = df_val_filtered["normalized_answer"].apply(lambda x: weight_dict[x])

print(
    f"Number of validation samples after filtering: {len(df_val_filtered)} ({len(df_val_filtered)/len(df_val)*100: .2f} % )"
)


#
#
#

### DATA PROCESSING ###

# Tokenizer
with open(config.folder_path/"tokenizer_word_index.json", "r", encoding="utf-8") as file:
    word_index = json.load(file)
tokenizer = Tokenizer(word_index=word_index, maxlen=config.maxlen)
num_words = len(word_index)

# GloVe embedding
glove_emb, _ = get_GloVe_emb(
    config.glove_path,
    dim=config.emb_dim,
    word_index=word_index
)

# Onehot encoding
list_train_gt = list(df_train_filtered["normalized_answer"])
train_gt = np.reshape(np.array(list_train_gt), (-1, 1))
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float32)
onehot_encoder.fit(train_gt)

# saving ordered possible answers
possible_ans = (onehot_encoder.categories_[0]).tolist()
with open(saving_folder/"possible_answers.json", "w") as f:
    json.dump(possible_ans, f)

# correspondence between student and teacher answers indices
answers_labels = {}
with open(config.KD_path / "answer2label.txt", encoding='utf-8') as file:
    for row in file:
        diz = json.loads(row.strip())
        answers_labels[diz['answer']] = diz['label']

teacher_indexes = []
for a in possible_ans:
    teacher_indexes.append(answers_labels[a])


# Train data loader
train_data = Custom_Generator(
    df_train_filtered, 
    config.dataset_path,
    tokenizer,
    onehot_encoder,
    logits_path=config.KD_path/"train_logits",
    indexes_to_consider=teacher_indexes,
    im_size=config.im_size,
    num_channels=config.num_channels,
    sample_weights=True,
    batch_size=BS_SIZE,
    shuffle=True,
)

# Vaild data loader
valid_data = Custom_Generator(
    df_val_filtered,
    config.dataset_path,
    tokenizer,
    onehot_encoder,
    logits_path=config.KD_path/"val_logits",
    indexes_to_consider=teacher_indexes,    
    im_size=config.im_size,
    num_channels=config.num_channels,
    sample_weights=True,
    batch_size=BS_SIZE,
    shuffle=False,
)

#
#
#


### MODEL ###
model = MFB_Baseline(
    k=config.k,
    maxlen=config.maxlen,
    num_words=num_words,
    emb_dim=config.emb_dim,
    glove_emb=glove_emb,
    im_size=config.im_size,
    num_channels=config.num_channels,
    num_classes=config.num_classes,
    dropout_rate=config.dropout_rate,
    last_softmax=False
)

print('Number of parameters:', model.count_params())
#print(model.summary())

#
#
#


### TRAINING ###

distiller = Distiller(student=model)

dummy_image = np.random.randint(0, 256, (1,config.im_size,config.im_size,config.num_channels))/255*2-1
dummy_answer = np.random.randint(0, num_words, (1,config.maxlen))
_ = distiller.predict((dummy_answer, dummy_image))

distiller.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    metrics=["accuracy"],
    student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=False),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=config.ALPHA,
    temperature=config.TEMPERATURE
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=False
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=saving_folder / "best_distiller.weights.h5",
        save_weights_only=True,
        save_best_only=True,
        monitor="val_loss",
        initial_value_threshold=0.1,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        # cooldown=2,
        min_lr=5e-5, 
        verbose=0,
    ),
]

history = distiller.fit(
    train_data, 
    validation_data=valid_data, 
    epochs=NUM_EPOCHS, 
    callbacks=callbacks
)

#
#
#


### SAVING ###

# saving loss and accuracy trend
train_history = history.history
with open(saving_folder / "training_history.json", "w") as f:
    json.dump(train_history, f)

# model saving
distiller.save_weights(saving_folder / "final_distiller.weights.h5")

model.compile(
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    metrics=["accuracy"],
)
model.save(saving_folder / "final_model.keras")

model2 = MFB_Baseline(
    k=config.k,
    maxlen=config.maxlen,
    num_words=num_words,
    emb_dim=config.emb_dim,
    glove_emb=glove_emb,
    im_size=config.im_size,
    num_channels=config.num_channels,
    num_classes=config.num_classes,
    dropout_rate=config.dropout_rate,
    last_softmax=False
)
distiller2 = Distiller(student=model2)
_ = distiller2.predict((dummy_answer, dummy_image))
distiller2.load_weights(saving_folder / "best_distiller.weights.h5")
model2.compile(
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    metrics=["accuracy"],
)
model2.save(saving_folder / "best_model.keras")

#
#
#

### PERFORMANCE ###

if True:

    train_data = Custom_Generator(
    df_train_filtered, 
    config.dataset_path,
    tokenizer,
    onehot_encoder,
    im_size=config.im_size,
    num_channels=config.num_channels,
    sample_weights=True,
    batch_size=BS_SIZE,
    shuffle=False,
    )
    valid_data = Custom_Generator(
    df_val_filtered,
    config.dataset_path,
    tokenizer,
    onehot_encoder,  
    im_size=config.im_size,
    num_channels=config.num_channels,
    sample_weights=True,
    batch_size=BS_SIZE,
    shuffle=False,
    )

    final_model = keras.models.load_model(saving_folder / "final_model.keras")

    try:
        best_model = keras.models.load_model(saving_folder / "best_model.keras")
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
    print(f'- Train Accuracy: {final_train_accuracy*100:.2f} %, Val Accuracy: {final_val_accuracy*100:.2f} %')

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
        print(f'- Train Accuracy: {best_train_accuracy*100:.2f} %, Val Accuracy: {best_val_accuracy*100:.2f} %')

    with open(saving_folder / "performance.json", "w") as file:
        json.dump(performance, file)