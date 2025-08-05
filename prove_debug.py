from pathlib import Path
import os

from main_train import get_args
from utils.config import load_config
from data.dataset import get_vqav2

config = load_config()
df = get_vqav2(config["paths"]["dataset_path"])
print("Columns: ", [i for i in df.columns])
print(df.head())
