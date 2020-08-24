from transformers.tokenization_auto import AutoTokenizer
import torch
import random
import pathlib

class Wellness:
    def __init__(self, dataroot='/mnt/data/pretrained/wellness'):
        dataroot = pathlib.Path(dataroot)

        self.answers = torch.load(dataroot/'answers.pt')
        self.label_map = torch.load(dataroot/'label_map.pt')

        model_name_or_path = "monologg/koelectra-base-discriminator"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.model = torch.load(dataroot/'wellness_model.pt').to('cpu')
        self.model.eval()

    def __call__(txts):
        inputs_ = tokenizer(txts, return_tensors='pt', padding='max_length', truncation=True)
        ixs = torch.argmax(self.model(**inputs_)[0], dim=-1).tolist()
        answers = [random.choice(self.answers.get(ix, [''])) for ix in ixs]
        labels = [self.label_map[ix] for ix in ixs]

        return answers, labels
