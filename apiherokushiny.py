# bert_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Carregue o modelo BERT (exemplo CPU-only)
model = SentenceTransformer('all-MiniLM-L6-v2')

class TextRequest(BaseModel):
    text: str

@app.post("/embed")
async def get_embedding(req: TextRequest):
    embedding = model.encode(req.text).tolist()
    return {"embedding": embedding}

@app.get("/")
async def root():
    return {"message": "BERT API running"}
