import sys

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from scipy.io.wavfile import write
import os
import io
sys.path.insert(0, '..')

app = FastAPI()
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

import wellness

def read_root():
    return {"Hello": "World"}

@app.get("/talk")
async def voice(q: str):
  questions = [q]
  answers, labels = wellnes(questions)
  return {"labels": labels, "answer": answers}
