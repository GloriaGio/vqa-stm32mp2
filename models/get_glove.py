import numpy as np


def get_GloVe_emb(GloVe_folder, dim=50, word_index=None):
    # Load GloVe embeddings:
    # all 400k if word_index is None, otherwise only for words in word_index

    path = GloVe_folder / f"glove.6B.{dim}d.txt"

    if word_index == None:
        word_index = {}
        with open(path, "r", encoding="utf-8") as file:
            for n, line in enumerate(file):
                values = line.strip().split()
                word_index[values[0]] = n

    emb_dict = {}

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            values = line.strip().split()
            if values[0] in word_index.keys():
                emb_dict[values[0]] = values[1:]

    emb = []
    sorted_wi = sorted(word_index.items(), key=lambda x: x[1])
    for word, i in sorted_wi:
        if word == "<pad>":
            emb.append([0] * dim)
        elif word not in emb_dict.keys():
            emb.append(emb_dict["<unk>"])
        else:
            emb.append(emb_dict[word])

    emb = np.array(emb, dtype="float32")

    return emb, word_index
