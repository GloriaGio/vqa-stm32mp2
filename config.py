from pathlib import Path

# paths
path = Path.home() / "Desktop" / "VQA"
# folder path
folder_path = path / "VQAforMCUs"
# dataset folder
dataset_path = path / "vqa_dataset"
# KD folder
KD_path = path / "KD"
# GloVe folder
glove_path = path / "glove.6B"


# input and output parameters
maxlen = 15  # maximum question length
min_freq = 5 # in the vocabolary words with freq > min_freq
im_size = 224  # image higth and width
num_channels = 3  # image channels
num_classes = 1000  # number of possible answers

# net parameters
k = 2
emb_dim = 50
dropout_rate = 0

# training parameters
NUM_EPOCHS = 2 #100  # epochs number
LR = 0.001  # learning rate
BS_SIZE = 32  # batch size

# knowledge distillation parameters
ALPHA = 0.1 # student loss weight (1-ALPHA is distillation loss weight)
TEMPERATURE = 3 # parameter of distillation loss