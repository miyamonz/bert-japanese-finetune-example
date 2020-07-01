import torch
from transformers import BertJapaneseTokenizer, BertForSequenceClassification, AdamW

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split


def get_tokenizer():
    return BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

# num_labels = len(categories)


def get_model(num_labels):
    model = BertForSequenceClassification.from_pretrained(
        'cl-tohoku/bert-base-japanese-whole-word-masking',
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False,
    )
    return model


def get_optimizers(model):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    return optimizer


def _get_datasets(*args):
    dataset = TensorDataset(*args)

    # ここらへんもっときれいにできそう
    # 90%地点のIDを取得
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # データセットを分割
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    return train_ds, val_ds


def get_dataloader(ds_source, batch_size):
    """source will be passed TensorDataset"""
    train_ds, valid_ds = _get_datasets(*ds_source)

    train_dl = DataLoader(
        train_ds,
        sampler=RandomSampler(train_ds),
        batch_size=batch_size
    )

    valid_dl = DataLoader(
        valid_ds,
        sampler=SequentialSampler(valid_ds),
        batch_size=batch_size
    )
    return train_dl, valid_dl
