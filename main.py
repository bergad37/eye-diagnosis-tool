from fastapi import FastAPI, UploadFile, File
import os
from scripts.predict import run_prediction
import uuid


app = FastAPI()

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    os.makedirs("temp", exist_ok=True)

    file_path = f"temp/{uuid.uuid4()}.jpg"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    result = run_prediction(file_path)

    os.remove(file_path)

    return result