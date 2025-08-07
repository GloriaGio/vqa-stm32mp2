def preliminary_performance(model, train_data, valid_data, verbose=True):
    # Evaluate the model's preliminary loss and standard accuracy

    train_loss, train_accuracy = model.evaluate(train_data)
    val_loss, val_accuracy = model.evaluate(valid_data)

    if verbose:
        print(f"Train Loss: {train_loss}, Val Loss: {val_loss}")
        print(f"Train Accuracy: {train_accuracy}, Val Accuracy: {val_accuracy}")

        return {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }


def vqa_accuracy(model_ans, list_10ans):
    # Calculate VQA v2 accuracy by comparing model answers to
    # the list of 10 ground-truth answers per question

    count = 0
    for i in range(len(model_ans)):
        count += min(list_10ans[i].count(model_ans[i]) / 3.0, 1.0)
    return count / len(model_ans)


def get_model_ans(model, dataloader, possible_ans):
    # Generate model predictions on the given dataloader and
    # convert predicted indices to answer strings

    model_ans = []
    for i in range(len(dataloader)):
        batch_input, _, _ = dataloader[i]
        pred = model.predict(batch_input, verbose=0)
        ans_idx = pred.argmax(axis=-1)
        batch_ans = [possible_ans[idx] for idx in ans_idx]
        model_ans += batch_ans

    return model_ans
