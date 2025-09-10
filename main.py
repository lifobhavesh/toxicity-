from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

# Model load
classifier = pipeline("text-classification", model="unitary/toxic-bert")

# FastAPI app
app = FastAPI(title="Toxicity API")

# CORS allow front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ya apne website domain ka URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input data model
class TextInput(BaseModel):
    text: str

# API endpoint
@app.post("/predict")
def predict_toxicity(data: TextInput):
    result = classifier(data.text)
    return {"result": result}
