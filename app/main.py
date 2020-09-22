import sys
sys.path.insert(0, '..')

import os
import io
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)

from wellness import KoGPT2Chat

model = KoGPT2Chat.load_from_checkpoint("/mnt/data/pretrained/wellness/chatbot.ckpt")

@app.get("/talk")
async def talk(q: str = "벽에 머리를 부딪히는 느낌이야"):
    logger.info(f"q={q}")
    ans = model.talk(q)
    #answers, labels = w([q])
    #return {"labels": labels, "answer": answers, "questions": [q], "text": answers[0]}
    return {"answer": ans, "questions": q, "text": ans}
