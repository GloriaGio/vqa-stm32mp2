from tensorflow import keras
import numpy as np

from pathlib import Path
from dataset import get_vqav2


def vqa_accuracy(model_ans, list_10ans):
    count = 0
    for i in range(len(model_ans)):
        count += min(list_10ans[i].count(model_ans[i]) / 3.0, 1.0)
    return count / len(model_ans)


if __name__ == "__main__":

    # path cartella del modello = ... (magari gliela faccio dare da linea di comando)

    dataset_path = (
        Path.home() / "Desktop" / "STMicroelectronics" / "VQA improved" / "vqa_dataset"
    )

    df_train = get_vqav2(dataset_path, train=True, keep_10ans=True, verbose=True)
    df_val = get_vqav2(dataset_path, train=False, keep_10ans=True, verbose=True)

    # carico tokenizer
    # uso custom generator senza logits

    # carico possibili risposte: possible_ans

    # carico modello
    # model = keras.models.load_model(folder / "PROVA.keras")

    # train_pred = model.predict(train_loader)
    # train_ans_idx = train_pred.argmax(axis=-1)
    # train_ans = [possible_ans[idx] for idx in train_ans_idx]

    # esempio ma con gt
    train_ans = list(df_train["normalized_answer"])[:10000]
    train_10ans = list(df_train["normalized_10answers"])[:10000]

    accuracy = vqa_accuracy(train_ans, train_10ans)
    print(accuracy)

    # val_pred = model.predict(val_loader)
    # val_ans_idx = val_pred.argmax(axis=-1)
    # val_ans = [possible_ans[idx] for idx in val_ans_idx]

    # salvo in un json nella stessa cartella di partenza
