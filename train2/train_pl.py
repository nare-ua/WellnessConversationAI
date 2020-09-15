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


import prep

class WellnessDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class WellnessDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.train_dims = None
        self.args = args

    def prepare_data(self):
        # called only on 1 GPU
        self.utters, self.label_map, self.answers = prep.load()
        self.num_labels = len(self.label_map)
        print("# labels: ", len(self.label_map))
        print("# utterence:", sum(len(v) for v in self.utters.values()))
        print("# answers: ", sum(len(v) for v in self.answers.values()))

    def setup(self, stage=None):
        # called on every GPU
        self.train, self.val = self.load_datasets()
        self.num_labels = len(self.label_map)
        #self.train_dims = self.train.next_batch.size()

    def load_datasets(self):
        model_name_or_path = "monologg/koelectra-base-discriminator"
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        utters, label_map, answers = self.utters, self.label_map, self.answers

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


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.args.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.train, batch_size=self.args.batch_size, drop_last=True)

class WellnessClassifier(pl.LightningModule):
    def __init__(self, hparams, num_labels):
        super().__init__()
        self.hparams = hparams
        model_name_or_path = "monologg/koelectra-base-discriminator"
        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        #inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}

        outputs = self(**batch)
        loss = outputs[0]

        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        tensorboard_logs = {"loss": loss, "rate": lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        #inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}

        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        targets = batch["labels"]
        return {'val_loss': val_loss, "logits": logits, "target": targets}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x['val_loss'] for x in outputs]).mean()
        preds = torch.stack([x['logits'] for x in outputs]).argmax(dim=-1)
        targets = torch.stack([x['target'] for x in outputs])
        acc = accuracy(preds, targets)

        log_dict = {'val_loss': loss_val, 'acc': acc}

        return {'log': log_dict, 'val_loss': log_dict['val_loss'], 'progress_bar': log_dict}

    def get_lr_scheduler(self, opt):
        scheduler = get_linear_schedule_with_warmup(
            opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.train_steps
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def setup(self, mode):
        if mode == "fit":
            #total_devices = self.hparams.gpus * self.hparams.n_nodes
            total_devices = self.hparams.gpus
            train_batches = len(self.train_dataloader()) // total_devices
            self.train_steps = (self.hparams.max_epochs * train_batches) // self.hparams.accumulate_grad_batches

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(
            optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
        )

        scheduler = self.get_lr_scheduler(opt)

        return [opt], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=5e-5)
        parser.add_argument('--adam_epsilon', type=float, default=1e-08)
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--num_train_epochs", dest="max_epochs", default=25, type=int)
        parser.add_argument(
            "--gradient_accumulation_steps",
            dest="accumulate_grad_batches",
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.",
        )
        parser.add_argument("--train_batch_size", default=32, type=int)
        return parser

def run_fit():
    import pprint
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = WellnessClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    args.gpus = 2
    args.max_epochs = 25
    args.train_batch_size = 8
    args.batch_size = 8

    pprint.pprint(vars(args))

    wellness_dm = WellnessDataModule(args)
    wellness_dm.prepare_data()

    model = WellnessClassifier(args, wellness_dm.num_labels)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=5,
        verbose=True,
        mode='min'
    )

    trainer = Trainer.from_argparse_args(args, early_stop_callback=early_stop_callback)
    trainer.fit(model, wellness_dm)

def test():
    import pprint
    import random
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = WellnessClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    model = WellnessClassifier(args, 359)
    ckpt_path = "lightning_logs/version_0/checkpoints/epoch=24.ckpt"
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt['state_dict'])

    model_name_or_path = "monologg/koelectra-base-discriminator"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    utters, label_map, answers = prep.load()
    label_map = {v:k for k, v in label_map.items()}

    samples = [
        "벽에 머리를 부딪히는 느낌이야",
        "허리가 아파서 움직임이 어렵네ㅠㅠ",
        "집중력도 떨어지고 기분이 좋지 않아",
        "나는 화가 통제가 안돼!",
        "히잉?",
        "나 자해 할거야",
        "팔다리가 너무 저려",
        "방에만 있고 싶어",
        "스트레스 너무 많이 받아서 잠이 안와",
        "난바부야 기억을 하나두 못하겠어",
        "다이어트 하고싶은데 맘처럼 안되네",
        "요즘은 이상한 생각이 많이 들어",
        "부정적인 생각이 많이 드네",
        "사고 휴유증이 있는걸까",
        "체력이 떨어져서 문제야",
        "으악! 꽥!",
        "요즘 비둘기 무서워",
        "감정이 왔다갔다해요.",
        "화가 많이 날때는 감정 조절이 안되어여",
        "요즘 잠이 안와요",
        "입맛도 통 없구",
        "기분이 우울해서 큰일이야",
        "나는 아무것도 잘한게 없는걸?",
        "모든걸 내 마음대로 하고 싶을 때 있잖아",
        "무엇이 불안한지 잘 모르겠어"
    ]
    model.eval()
    inputs_ = tokenizer(samples, return_tensors='pt', padding='max_length', truncation=True)
    ixs = torch.argmax(model(**inputs_)[0], dim=-1).tolist()
    res = [random.choice(answers.get(ix, ['None'])) for ix in ixs]
    labels = [label_map[ix] for ix in ixs]

    for s, l, r in zip(samples, labels, res):
        print(f"{l}|{s}=>{r}")

if __name__ == "__main__":
    test()
