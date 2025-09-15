import json

import cv2
import numpy as np
from tensorflow import keras
from data.onehot_encoder import OneHotEncoder

from data.text_processing import Tokenizer

#
#
#


class Custom_Generator(keras.utils.Sequence):

    def __init__(
        self,
        dataframe,
        data_path,
        tokenizer,
        onehot_encoder=None,
        logits_path=None,
        indexes_to_consider=None,
        im_size=224,
        num_channels=3,
        sample_weights=False,
        batch_size=32,
        shuffle=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_path = data_path
        self.logits_path = logits_path
        self.indexes_to_consider = np.array(indexes_to_consider)
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.onehot_encoder = onehot_encoder
        self.im_size = im_size
        self.num_channels = num_channels
        self.sample_weights = sample_weights
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.datalen = len(dataframe)
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return (np.ceil(self.datalen / float(self.batch_size))).astype(int)

    def __getitem__(self, idx):
        batch_indexes = self.indexes[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_df = self.dataframe.iloc[batch_indexes]

        # Get weights
        if self.sample_weights:
            batch_w = np.array(batch_df["weight"], dtype=np.float32)
        else:
            batch_w = np.ones(len(batch_df))

        # Tokenize questions
        batch_q = self.tokenizer.texts_to_sequences(batch_df["question"])

        # Load and preprocess images:
        # convert to RGB, resize to 224x224, and scale pixel values to [-1, 1]
        im_names = list(batch_df["image_name"])
        if "train" in im_names[0]:
            direct = "train2014"
        elif "val" in im_names[0]:
            direct = "val2014"
        elif "test" in im_names[0]:
            direct = "test2015"
        batch_im = []
        for im_name in im_names:
            if self.num_channels == 1:
                how = cv2.IMREAD_GRAYSCALE
                how2 = cv2.COLOR_GRAY2RGB
            else:
                how = cv2.IMREAD_COLOR
                how2 = cv2.COLOR_BGR2RGB
            im_path = self.data_path / direct / im_name
            im = cv2.imread(im_path, how)
            im = cv2.cvtColor(im, how2)
            im = cv2.resize(im, (self.im_size, self.im_size))
            im = im.astype(dtype="float32") / 255.0 * 2 - 1
            batch_im.append(im)
        batch_im = np.array(batch_im)

        # If no one-hot encoder is provided, return dummy zero vectors as ground truth
        if self.onehot_encoder is None:
            batch_gt = np.zeros((len(batch_df), 1000))
            return (batch_q, batch_im), batch_gt, batch_w

        # Apply one-hot encoding to the normalized answers
        batch_gt = list(batch_df["normalized_answer"])
        batch_gt = self.onehot_encoder.transform(batch_gt)

        # If no logits are provided, return inputs and ground truth;
        if self.logits_path is None:
            return (batch_q, batch_im), batch_gt, batch_w

        # Load teacher logits
        questions_id = batch_df["question_id"]
        logits = []
        for q_id in questions_id:
            logits_file = str(q_id) + ".json"
            with open(self.logits_path / logits_file, "r") as file:
                logits_dict = json.load(file)
                logits.append(logits_dict["logits"])
        logits = np.array(logits, dtype=np.float32)
        logits = logits[:, self.indexes_to_consider]

        return (batch_q, batch_im), (batch_gt, logits), batch_w

    def on_epoch_end(self):
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            np.random.shuffle(self.indexes)


#
#
#


def get_custom_generators(
    df_train, df_val, tokenizer_path, possible_ans_path, config, get_logits=True
):
    # Load tokenizer
    with open(tokenizer_path) as file:
        word_index = json.load(file)
    tokenizer = Tokenizer(word_index=word_index, maxlen=config["model"]["max_length"])

    # Set the final number of words in the vocabulary
    config["model"]["num_vocab_words"] = len(word_index)

    # Load onehot encoder
    with open(possible_ans_path) as file:
        possible_ans = json.load(file)
    onehot_encoder = OneHotEncoder(categories=possible_ans)

    # If knowledge distillation is enabled and logits are needed:
    # load teacher answer-to-label mapping, compute index correspondence between student and teacher answers,
    # and set paths to teacher's train/val logits. Otherwise, set values to None.
    if config["training"]["knowledge_distillation"] and get_logits:
        # Load teacher answers
        answers_labels = {}
        with open(
            config["paths"]["KD_path"] / "answer2label.txt", encoding="utf-8"
        ) as file:
            for row in file:
                diz = json.loads(row.strip())
                answers_labels[diz["answer"]] = diz["label"]

        # Create a list mapping student answer indices to corresponding teacher answer indices
        teacher_indexes = []
        for a in possible_ans:
            teacher_indexes.append(answers_labels[a])

        train_logits_path = config["paths"]["KD_path"] / "train_logits"
        val_logits_path = config["paths"]["KD_path"] / "val_logits"
    else:
        teacher_indexes = None
        train_logits_path = None
        val_logits_path = None

    # Train data loader
    train_data = Custom_Generator(
        df_train,
        config["paths"]["dataset_path"],
        tokenizer,
        onehot_encoder,
        logits_path=train_logits_path,
        indexes_to_consider=teacher_indexes,
        im_size=config["model"]["image_size"],
        num_channels=config["model"]["num_channels"],
        sample_weights=True,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )

    # Val data loader
    valid_data = Custom_Generator(
        df_val,
        config["paths"]["dataset_path"],
        tokenizer,
        onehot_encoder,
        logits_path=val_logits_path,
        indexes_to_consider=teacher_indexes,
        im_size=config["model"]["image_size"],
        num_channels=config["model"]["num_channels"],
        sample_weights=True,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )

    return train_data, valid_data
