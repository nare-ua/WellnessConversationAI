from argparse import ArgumentParser

from transformers.tokenization_auto import AutoTokenizer
from transformers.modeling_auto import AutoModelForSequenceClassification
from transformers import AutoConfig
from transformers.optimization import get_linear_schedule_with_warmup
from itertools import islice
from openpyxl import load_workbook
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping
import torch
from torch.optim import AdamW
from pytorch_lightning.metrics.functional import accuracy
from transformers.tokenization_auto import AutoTokenizer
import torch
import random

model_name_or_path = "monologg/koelectra-base-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

def load_datasets(self):

    utters, label_map, answers = prep.load()

    def gen_datasets(split=0.1):
        train_dataset, test_dataset = [], []
        train_label, test_label = [], []

        for l, u in utters.items():
            assert len(u) > 1, "# of utterences per label should be > 1"
            labels = [l] * len(u)
            ix = max(1, int(len(u)*split))
            test_dataset.extend(u[:ix])
            test_label.extend(labels[:ix])
            train_dataset.extend(u[ix:])
            train_label.extend(labels[ix:])
        return train_dataset, train_label, test_dataset, test_label

    train_dataset, train_label, test_dataset, test_label = gen_datasets()

    train_dataset = tokenizer(train_dataset, return_tensors='pt', padding='max_length', truncation=True)
    test_dataset = tokenizer(test_dataset, return_tensors='pt', padding='max_length', truncation=True)

    train_dataset['labels'] = train_label
    test_dataset['labels'] =  test_label

    keys = list(train_dataset.keys())
    train_dataset = [dict(zip(keys, v)) for v in zip(*train_dataset.values())]
    test_dataset = [dict(zip(keys, v)) for v in zip(*test_dataset.values())]

    train_dataset = WellnessDataset(train_dataset)
    test_dataset = WellnessDataset(test_dataset)

    return train_dataset, test_dataset

_, test_dataset = gen_datasets()

test_dataloader =  DataLoader(self.train, batch_size=self.args.batch_size, drop_last=True)

fn = 'wellness_model.pt'
model = torch.load(fn)
model.to('cpu')
model.eval()
