from fastapi import FastAPI, UploadFile, File
import os
from scripts.predict import run_prediction

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = f"temp/{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    result = run_prediction(file_path)

    os.remove(file_path)

    return result