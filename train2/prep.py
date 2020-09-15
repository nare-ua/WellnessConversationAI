import torch
from openpyxl import load_workbook
from itertools import islice
import collections
from pathlib import Path

def load(rootdir=Path(".")):
    fns = ['utters.pt', 'label_map.pt', 'answers.pt']
    if any(not (rootdir/fn).exists() for fn in fns):
        return parse_data()
    return [torch.load(p) for p in fns]


def parse_data():
    fn = "wellness.xlsx"
    wb = load_workbook(filename=fn)
    ws = wb[wb.sheetnames[0]]

    utters = collections.defaultdict(list)
    answers = collections.defaultdict(list)
    label_map = {}

    for row in islice(ws.iter_rows(), 1, None):
        label = row[0].value.strip()
        utter = row[1].value.strip()

        label_map.setdefault(label, len(label_map))
        utters[label_map[label]].append(utter)

        if row[2].value is not None:
            answers[label_map[label]].append(row[2].value.strip())

    torch.save(utters, 'utters.pt')
    torch.save(label_map, 'label_map.pt')
    torch.save(answers, 'answers.pt')

    return utters, label_map, answers
