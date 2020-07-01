import torch
import numpy as np

import time
from util import format_time, calc_time


# send batch to model and return loss value
def batch_loss(model, batch, opt=None):
    # batch should be input_ids, masks, and labes
    (input_ids, attention_mask, labels) = map(
        lambda x: x.to(model.device), batch)

    # this block depends on the model return types.
    loss, logits = model(input_ids,
                         attention_mask=attention_mask,
                         labels=labels)

    # if this function get opt argument, we update parameters
    if opt is not None:
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

    # Move logits to CPU
    logits = logits.detach().cpu().numpy()
    return loss.item(), logits

# Function to calculate the accuracy of our predictions vs labels


def flat_accuracy(preds, labels):
    pred_flat = preds.flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# 訓練パートの定義


def train(model, dl, optimizer):
    model.train()
    total_loss = 0
    for batch in dl:
        loss, _ = batch_loss(model, batch, optimizer)
        total_loss += loss
    return total_loss

# テストパートの定義


def validation(model, dl):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for batch in dl:
            loss, logits = batch_loss(model, batch)
            total_loss += loss

            preds = np.argmax(logits, axis=1)
            label_ids = batch[2].numpy()
            total_accuracy += flat_accuracy(preds, label_ids)
    return total_loss, total_accuracy

# TOOO: schedule周りがまだ入ってない
# from transformers import get_linear_schedule_with_warmup


def fit(model, train_dl, valid_dl, optimizer, epochs):
    training_stats = []
    total_t0 = time.time()

    device = torch.device("cuda")
    model.to(device)

    for epoch in range(epochs):
        print("")
        print(f'======== Epoch {epoch+1} / {epochs} ========')

        print('Training...')
        with calc_time() as done:
            total_train_loss = train(model, train_dl, optimizer)
            training_time = done()
        print(f"  Training took: {training_time}")
        avg_train_loss = total_train_loss / len(train_dl)

        print("")
        print("Running Validation...")
        with calc_time() as done:
            total_val_loss, total_accuracy = validation(model, valid_dl)
            validation_time = done()
        print(f"  Validation took: {validation_time}")
        avg_val_loss = total_val_loss / len(valid_dl)
        avg_val_accuracy = total_accuracy / len(valid_dl)

        training_stats.append({
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        })

    print("")
    print("Training complete!")
    print(f"Total training took {format_time(time.time()-total_t0)} (h:mm:ss)")

    return training_stats
