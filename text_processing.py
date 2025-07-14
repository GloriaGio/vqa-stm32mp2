from dataset import get_vqav2
import config

from collections import Counter
import re
import json
import numpy as np
from pathlib import Path


class Tokenizer:
    def __init__(
            self,  
            num_words=None, 
            min_freq=0,
            pad_token="<pad>",
            oov_token="<unk>", 
            word_index=None,
            maxlen=None
            ):
        self.num_words = num_words
        self.min_freq = min_freq
        self.maxlen = maxlen
        self.word_counts = {}

        if word_index is None:
            self.oov_token = oov_token
            self.pad_token = pad_token
            self.word_index = {}
            self.index_word = {}
            self.vocab_size = 0
            
        else:
            self.word_index = word_index
            self.index_word = {idx: word for word, idx in word_index.items()}
            self.oov_token = self.index_word[1]
            self.pad_token = self.index_word[0]
            self.vocab_size = len(word_index)
        
    def fit_on_texts(self, texts):
        if len(self.word_index) != 0:
            raise KeyError("word_index dictionary already exists")

        counter = Counter()
        for text in texts:
            clean_text = re.sub(r"\W+\s*", " ", text)
            clean_text = re.sub(r"_+\s*", " ", clean_text)
            tokens = clean_text.lower().split()
            counter.update(tokens)
        self.word_counts = counter.most_common()

        if self.num_words is None:
            most_common = self.word_counts
        else:
            most_common = counter.most_common(self.num_words-2)
        self.word_index = {word: idx + 2 for idx, (word, f) in enumerate(most_common) if f >= self.min_freq} 
        self.word_index[self.pad_token] = 0        
        self.word_index[self.oov_token] = 1
        self.index_word = {idx: word for word, idx in self.word_index.items()}
        self.vocab_size = len(self.word_index) 

    def texts_to_sequences(self, texts, dtype="int32"):
        sequences = []
        for text in texts:
            clean_text = re.sub(r"\W+\s*", " ", text)
            clean_text = re.sub(r"_+\s*", " ", clean_text)
            tokens = clean_text.lower().split()
            seq = [self.word_index.get(token, self.word_index[self.oov_token]) for token in tokens]
            if self.maxlen is not None:
                if len(seq) < self.maxlen:
                    pad = self.maxlen -len(seq)
                    seq = seq + [0]*pad
                else:
                    seq = seq[:self.maxlen]
            sequences.append(seq)
        return np.array(sequences, dtype=dtype)
    
    def save_json(self, saving_path):
        with open(saving_path, "w") as f:
            json.dump(self.word_index, f)




def get_GloVe_emb(GloVe_folder, dim=50, word_index=None):
    # word_index=None to get all 400k word embeddings
    # otherwise the embedding of words in word_index
    
    path = GloVe_folder/f"glove.6B.{dim}d.txt"
    
    if word_index == None:
        word_index = {}
        with open(path, 'r', encoding='utf-8') as file:
            for n, line in enumerate(file):
                values = line.strip().split()
                word_index[values[0]] = n
        
    emb_dict = {}
    
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.strip().split()
            if values[0] in word_index.keys():
                emb_dict[values[0]] = values[1:]
                
    emb = []
    sorted_wi = sorted(word_index.items(), key=lambda x:x[1])
    for word, i in sorted_wi:
        if word == '<pad>':
            emb.append([0]*dim)
        elif word not in emb_dict.keys():
            emb.append(emb_dict['<unk>'])
        else:
            emb.append(emb_dict[word])
                
    emb = np.array(emb, dtype='float32')
   
    return emb, word_index




    

if __name__ == "__main__":

    print("Loading the training data...")
    df_train = get_vqav2(config.dataset_path, train=True, keep_10ans=False, verbose=True)
    questions_list = list(df_train['question'])

    tok = Tokenizer(min_freq=config.min_freq, 
                    pad_token="<pad>",
                    oov_token="<unk>")
    tok.fit_on_texts(questions_list)
    print(tok.vocab_size)

    tok.save_json(config.folder_path/"tokenizer_word_index.json")