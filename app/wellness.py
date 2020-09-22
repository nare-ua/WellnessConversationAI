from transformers.tokenization_auto import AutoTokenizer
from transformers import AutoConfig
import torch
import random
import pathlib
from argparse import ArgumentParser
from gluonnlp.data import SentencepieceTokenizer
from transformers.modeling_auto import AutoModelForSequenceClassification
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from kogpt2.utils import get_tokenizer
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '<s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'

class KoGPT2Chat(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoGPT2Chat, self).__init__()
        self.hparams = hparams
        self.tok_path = get_tokenizer()
        self.neg = -1e18
        self.kogpt2, self.vocab = get_pytorch_kogpt2_model()
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max-len',
                            type=int,
                            default=32,
                            help='max sentence length on input (default: 32)')

        parser.add_argument('--batch-size',
                            type=int,
                            default=96,
                            help='batch size for training (default: 96)')
        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')
        return parser

    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        output, _ = self.kogpt2(inputs)
        return output

    def training_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        tensorboard_logs = {'train_loss': loss_avg}
        return {'loss': loss_avg, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        tensorboard_logs = {'val_loss': loss_avg}
        return {'val_loss': loss_avg, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        data, mask, label = zip(*batch)
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def prepare_data(self):
        datasetdir = Path('/mnt/data/datasets/wellness')
        fn = datasetdir/"wellness.xlsx"
        wb = load_workbook(filename=fn)
        ws = wb[wb.sheetnames[0]]
        wellness_data = pd.DataFrame(columns=["Q", "A", "label"])
        answers = {}
        for i, row in enumerate(ws.iter_rows()):
            label = row[0].value.strip()
            if label in ['배경/부모/죽음', '일반대화', '현재상태/증상감소', '현재상태/증상지속']:
                continue

            #if label == "현재상태/증상지속": continue
            utter = row[1].value.strip()
            wellness_data.loc[i] = [utter, None, label]
            answers.setdefault(label, [])
            if row[2].value is not None:
                answers[label].append(row[2].value.strip())

        self.wellness_answers = answers
        wellness_data["type"] = "wellness"

        self.data = pd.read_csv('Chatbot_data/ChatbotData.csv')
        self.data["type"] = "chat"
        self.data = self.data.append(wellness_data)
        self.data["set"] = np.random.choice(["train", "val"], p =[.9, .1], size=(self.data.shape[0],))

    def train_dataloader(self):
        self.train_set = CharDataset(self.data[self.data.set=="train"], self.tok_path, self.vocab, max_len=self.hparams.max_len,
                                     wellness_answers=self.wellness_answers)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=2,
            shuffle=True, collate_fn=self._collate_fn)
        return train_dataloader

    def val_dataloader(self):
        self.val_set = CharDataset(self.data[self.data.set=="val"], self.tok_path, self.vocab,
                                   max_len=self.hparams.max_len, wellness_answers=self.wellness_answers)
        val_dataloader = DataLoader(
            self.val_set, batch_size=self.hparams.batch_size, num_workers=2,
            shuffle=False, collate_fn=self._collate_fn)
        return val_dataloader

    def chat(self, sent='0'):
        tok = SentencepieceTokenizer(self.tok_path, num_best=0, alpha=0)
        sent_tokens = tok(sent)
        with torch.no_grad():
            while 1:
                q = input('user > ').strip()
                if q == 'quit':
                    break
                q_tok = tok(q)
                a = ''
                a_tok = []
                while 1:
                    input_ids = torch.LongTensor(
                        [self.vocab[U_TKN]] + self.vocab[q_tok] + [self.vocab[EOS]]+
                        #self.vocab[EOS, SENT] + self.vocab[sent_tokens] +
                        #self.vocab[EOS, S_TKN] +
                        [self.vocab[S_TKN]] +
                        self.vocab[a_tok]).unsqueeze(dim=0)
                    pred = self(input_ids)
                    gen = self.vocab.to_tokens(
                        torch.argmax(
                            pred,
                            dim=-1).squeeze().numpy().tolist())[-1]
                    if gen == EOS:
                        break
                    a += gen.replace('▁', ' ')
                    a_tok = tok(a)
                print("Simsimi > {}".format(a.strip()))

    def talk(self, q):
        tok = SentencepieceTokenizer(self.tok_path, num_best=0, alpha=0)
        #sent_tokens = tok(sent)
        q_tok = tok(q)
        a = ''
        a_tok = []
        while 1:
            input_ids = torch.LongTensor(
                [self.vocab[U_TKN]] + self.vocab[q_tok] + [self.vocab[EOS]]+
                #self.vocab[EOS, SENT] + self.vocab[sent_tokens] +
                #self.vocab[EOS, S_TKN] +
                [self.vocab[S_TKN]] +
                self.vocab[a_tok]).unsqueeze(dim=0)
            pred = self(input_ids)
            gen = self.vocab.to_tokens(
                torch.argmax(
                    pred,
                    dim=-1).squeeze().numpy().tolist())[-1]
            if gen == EOS:
                break
            a += gen.replace('▁', ' ')
            a_tok = tok(a)
        return a.strip()


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

class WellnessGPT2(pl.LightningModule):
    def __init__(self, hparams, num_labels):
        super().__init__()
        self.hparams = hparams
        model_name_or_path = "monologg/koelectra-base-discriminator"
        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)

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

