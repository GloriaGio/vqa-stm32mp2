from tensorflow import keras
import numpy as np
import cv2
import json

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

        # weights
        if self.sample_weights:
            batch_w = np.array(batch_df["weight"], dtype=np.float32)
        else:
            batch_w = np.ones(len(batch_df))

        # questions
        batch_q = self.tokenizer.texts_to_sequences(batch_df["question"])

        # images
        im_names = list(batch_df["image_name"])
        if "train2014" in im_names[0]:
            direct = "train2014"
        else:
            direct = "val2014"
        batch_im = []
        for im_name in im_names:
            if self.num_channels == 1:
                how = cv2.IMREAD_GRAYSCALE
            else:
                how = cv2.IMREAD_COLOR
            im_path = self.data_path / direct / im_name
            im = cv2.imread(im_path, how)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (self.im_size, self.im_size))
            if self.num_channels == 1:
                im = np.expand_dims(im, axis=-1)
            im = im.astype(dtype="float32") / 255.0 * 2 - 1
            batch_im.append(im)
        batch_im = np.array(batch_im)

        if self.onehot_encoder is None:
            batch_gt = np.zeros((len(batch_df), 1000))
            return (batch_q, batch_im), batch_gt, batch_w

        # onehot encoding
        batch_gt = list(batch_df["normalized_answer"])
        batch_gt = np.reshape(np.array(batch_gt), (-1, 1))
        batch_gt = self.onehot_encoder.transform(batch_gt)

        if self.logits_path is None:
            return (batch_q, batch_im), batch_gt, batch_w

        # logits
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
