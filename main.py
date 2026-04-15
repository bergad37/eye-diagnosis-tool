from fastapi import FastAPI, UploadFile, File
import os
import uuid

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    """
    Render expects the process to bind to $PORT quickly.
    Model loading can be slow and/or require files that aren't present at deploy time,
    so we keep startup non-blocking and non-fatal by default.
    """
    preload = os.getenv("PRELOAD_MODELS", "0").lower() in {"1", "true", "yes"}
    if not preload:
        return

    try:
        from scripts.predict import load_models
        load_models()
    except Exception as e:
        # Don't crash the web process; prediction endpoint will surface errors if any.
        print(f"⚠️ Model preload skipped due to error: {e}")


@app.get("/")
def home():
    return {"status": "running"}


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    import asyncio
    os.makedirs("temp", exist_ok=True)

    file_path = f"temp/{uuid.uuid4()}.jpg"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Import here so the web server can start/bind quickly even if
    # ML dependencies are heavy or fail to load at boot time.
    from scripts.predict import run_prediction
    # Run CPU-heavy work off the event loop to avoid the server appearing hung.
    result = await asyncio.to_thread(run_prediction, file_path)

    os.remove(file_path)

    return result