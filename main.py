import io
import time
import base64
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from gradcam_module.inference import GradCAMInference
from report.pdf_generator import build_pdf

# ── Startup ─────────────────────────────────────────
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("⏳ Loading model...")
    pipeline = GradCAMInference()
    print("✅ Model ready")
    yield
    print("🛑 Shutting down")

# ── App ─────────────────────────────────────────────
app = FastAPI(
    title="Medical AI — Chest X-Ray API",
    description="DenseNet121 + Grad-CAM for 14 pulmonary pathologies",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Routes ───────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "densenet121-res224-nih",
        "pathologies": 14,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    start = time.time()
    try:
        results = pipeline.predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    elapsed = round(time.time() - start, 2)

    sorted_results = dict(
        sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)
    )

    return {
        "filename": file.filename,
        "inference_time_sec": elapsed,
        "predictions": sorted_results
    }


@app.post("/report")
async def generate_report(request: Request):
    """
    Accepts JSON body with patient info + predictions + original image.
    Returns a PDF file.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    try:
        pdf_bytes = build_pdf(
            patient_name   = body.get("name",     "N/A"),
            patient_age    = str(body.get("age",  "N/A")),
            patient_gender = body.get("gender",   "N/A"),
            patient_id     = body.get("pid",      "N/A"),
            doctor         = body.get("doctor",   "N/A"),
            notes          = body.get("notes",    ""),
            filename       = body.get("filename", "unknown"),
            predictions    = body.get("predictions", {}),
            orig_b64       = body.get("orig_b64", ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    patient_name = body.get("name", "report").replace(" ", "_")
    date_str     = datetime.now().strftime("%Y%m%d_%H%M")
    fname        = f"xray_report_{patient_name}_{date_str}.pdf"

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={fname}"}
    )