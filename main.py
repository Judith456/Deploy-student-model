# main.py
from fastapi import (
    FastAPI, File, UploadFile,
    HTTPException)
from fastapi.middleware.cors import (
    CORSMiddleware)
from fastapi.responses import JSONResponse
import tempfile, os, time
from predictor_attention import predict

app = FastAPI(
    title   = "Cycling Form API",
    description =
        "Attention Student Model — 98.38%",
    version = "2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"]
)

@app.get("/")
def home():
    return {
        "message": "🚴 Cycling Form API",
        "model":   "Attention Student",
        "accuracy":"98.38%",
        "params":  "9,379",
        "size":    "35 KB",
        "endpoints":{
            "predict": "POST /predict",
            "health":  "GET  /health",
            "docs":    "GET  /docs"
        }
    }

@app.get("/health")
def health():
    return {
        "status":   "healthy",
        "model":    "Attention Student",
        "accuracy": "98.38%",
        "params":   9379,
        "classes":  ["Poor","Average","Good"]
    }

@app.post("/predict")
async def predict_video(
    file: UploadFile = File(...)):

    allowed = [".mp4",".avi",
               ".mov",".mkv"]
    ext     = os.path.splitext(
        file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Use: {allowed}")

    if file.size and file.size > 20_000_000:
        raise HTTPException(
            status_code=400,
            detail="File too large (max 20MB)")

    with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        start  = time.time()
        result = predict(tmp_path)
        result["processing_time"] = round(
            time.time()-start, 2)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)