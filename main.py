from fastapi import FastAPI, UploadFile, File
import os
import uuid
from scripts.predict import run_prediction

app = FastAPI()


# ✅ ADD IT HERE (right after app = FastAPI())
@app.on_event("startup")
def startup_event():
    print("🔥 Preloading models...")

    from scripts.predict import load_models
    load_models()

    print("✅ Models ready")


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