from transformers.tokenization_auto import AutoTokenizer
from transformers import AutoConfig
import torch
import random
import pathlib
from argparse import ArgumentParser
from transformers.modeling_auto import AutoModelForSequenceClassification
import pytorch_lightning as pl

class Wellness:
    def __init__(self, dataroot='/mnt/data/pretrained/wellness'):
        dataroot = pathlib.Path(dataroot)

        self.answers = torch.load(dataroot/'answers.pt')
        self.label_map = torch.load(dataroot/'label_map.pt')

        model_name_or_path = "monologg/koelectra-base-discriminator"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.model = torch.load(dataroot/'wellness_model.pt', map_location=torch.device('cpu'))
        self.model.eval()

    def __call__(self, txts):
        inputs_ = self.tokenizer(txts, return_tensors='pt', padding='max_length', truncation=True)
        ixs = torch.argmax(self.model(**inputs_)[0], dim=-1).tolist()
        answers = [random.choice(self.answers.get(ix, [''])) for ix in ixs]
        labels = [self.label_map[ix] for ix in ixs]

        return answers, labels

class WellnessClassifier(pl.LightningModule):
    def __init__(self, hparams, num_labels):
        super().__init__()
        self.hparams = hparams
        model_name_or_path = "monologg/koelectra-base-discriminator"
        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)

    def forward(self, **inputs):
        return self.model(**inputs)

class WellnessPL:
    def __init__(self, dataroot='/mnt/data/pretrained/wellness'):
        dataroot = pathlib.Path(dataroot)

        self.answers = torch.load(dataroot/'answers.pt')
        self.label_map = torch.load(dataroot/'label_map.pt')

        model_name_or_path = "monologg/koelectra-base-discriminator"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        #self.model = torch.load(dataroot/'wellness_model.pt', map_location=torch.device('cpu'))
        #self.model.eval()

        #parser = ArgumentParser()
        #args = parser.parse_args()

        model = WellnessClassifier({}, 359)
        ckpt_path = dataroot/"epoch=24.ckpt"
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        del ckpt['state_dict']['model.electra.embeddings.position_ids']
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        self.model = model

    def __call__(self, txts):
        inputs_ = self.tokenizer(txts, return_tensors='pt', padding='max_length', truncation=True)
        ixs = torch.argmax(self.model(**inputs_)[0], dim=-1).tolist()
        answers = [random.choice(self.answers.get(ix, [''])) for ix in ixs]
        labels = [self.label_map[ix] for ix in ixs]

        return answers, labels

