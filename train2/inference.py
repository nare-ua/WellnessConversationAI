from transformers.tokenization_auto import AutoTokenizer
import torch
import random


answers = torch.load('answers.pt')
label_map = torch.load('label_map.pt')

model_name_or_path = "monologg/koelectra-base-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

fn = 'wellness_model.pt'
model = torch.load(fn)

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
model.to('cpu')
model.eval()

inputs_ = tokenizer(samples, return_tensors='pt', padding='max_length', truncation=True)
ixs = torch.argmax(model(**inputs_)[0], dim=-1).tolist()
res = [random.choice(answers.get(ix, ['None'])) for ix in ixs]
labels = [label_map[ix] for ix in ixs]

for s, l, r in zip(samples, labels, res):
    print(f"{l}|{s}=>{r}")


