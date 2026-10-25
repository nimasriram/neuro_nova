from fastapi.staticfiles import StaticFiles
import io
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

# ---------- Configuration ----------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "hr_daya.keras"
STATIC_DIR = BASE_DIR / "static"
IMG_HEIGHT, IMG_WIDTH = 200, 300          # size the model was trained on
MAX_UPLOAD_BYTES = 5 * 1024 * 1024        # reject files over 5 MB

# Order must match the training labels (alphabetical from LabelEncoder)
EMOTIONS = {"A": "Angry", "F": "Fearful", "H": "Happy", "N": "Neutral"}
CLASS_CODES = list(EMOTIONS)

# ---------- Response models (these drive the docs) ----------
class Prediction(BaseModel):
    label: str = Field(description="Emotion code with the highest probability.", examples=["N"])
    emotion: str = Field(description="Readable name of the predicted emotion.", examples=["Neutral"])
    confidence: float = Field(ge=0, le=1, description="Probability of the predicted emotion (0 to 1).", examples=[0.82])
    probabilities: dict[str, float] = Field(
        description="Probability for each emotion code.",
        examples=[{"A": 0.05, "F": 0.04, "H": 0.09, "N": 0.82}],
    )

class Health(BaseModel):
    status: str = Field(examples=["ok"])

# ---------- Dark theme for the /docs page (black background, white text, light blue accent) ----------
DARK_CSS = """
:root { color-scheme: dark; }
html, body { background: #04060c !important; }
.swagger-ui, .swagger-ui * { color: #fff; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif !important; }
.swagger-ui a { color: #7dd3fc; }
.swagger-ui .scheme-container,
.swagger-ui .opblock-body,
.swagger-ui .opblock .opblock-section-header,
.swagger-ui .responses-inner,
.swagger-ui section.models,
.swagger-ui .model-box,
.swagger-ui .dialog-ux .modal-ux,
.swagger-ui table thead tr th { background: #04060c !important; }
.swagger-ui input, .swagger-ui textarea, .swagger-ui select {
    background: #0b1120 !important; color: #fff !important; border: 1px solid #3b6a80 !important;
}
.swagger-ui .markdown code, .swagger-ui .renderedMarkdown code { background: #0b1120; }
.swagger-ui svg { fill: #fff; }
.swagger-ui .btn { border-color: #7dd3fc; }
.swagger-ui .btn.execute { background: #7dd3fc !important; border-color: #7dd3fc !important; color: #04121c !important; }
.swagger-ui .opblock.opblock-get,
.swagger-ui .opblock.opblock-post { border-color: #7dd3fc; background: rgba(125, 211, 252, 0.07); }
.swagger-ui .opblock.opblock-get .opblock-summary-method,
.swagger-ui .opblock.opblock-post .opblock-summary-method { background: #7dd3fc; color: #04121c; }
.swagger-ui .opblock.opblock-get .opblock-summary,
.swagger-ui .opblock.opblock-post .opblock-summary { border-color: #7dd3fc; }
"""

# ---------- App and docs metadata ----------
tags_metadata = [
    {"name": "Prediction", "description": "Classify the emotion shown in a face image."},
    {"name": "System", "description": "Service status."},
]

app = FastAPI(
    title="HR.DAYA Emotion Classification API",
    summary="Classifies facial emotion with a CNN trained on the Chicago Face Database.",
    description="""
Serves the HR.DAYA model from the Voyager Wellness (NASA HUNCH) capstone project.

## How to use it
1. Open **POST /predict** below and click **Try it out**.
2. Upload a front-facing face image (`.jpg` or `.png`, up to 5 MB).
3. Click **Execute** and read the **Response body**.

## About the model
- 4 classes: **A**ngry, **F**earful, **H**appy, **N**eutral
- Input is resized to 300 x 200 pixels and scaled to 0 to 1
- Trained on posed studio photos, so accuracy on real-world images will be lower

> Research prototype. Not intended for medical or mental-health decisions.
""",
    version="1.0.0",
    openapi_tags=tags_metadata,
    docs_url=None,    # replaced by the custom dark /docs route below
    redoc_url=None,
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,                # hide the "Schemas" section
        "syntaxHighlight": {"theme": "monokai"},
    },
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

model = tf.keras.models.load_model(MODEL_PATH)

# ---------- Pages ----------
@app.get("/", include_in_schema=False)
def home():
    """Landing page with instructions and an in-browser analyzer."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/docs", include_in_schema=False)
def custom_docs():
    page = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Docs",
        swagger_ui_parameters=app.swagger_ui_parameters,
    )
    html = page.body.decode().replace("</head>", f"<style>{DARK_CSS}</style></head>")
    return HTMLResponse(html)

# ---------- API endpoints ----------
@app.get("/health", response_model=Health, tags=["System"], summary="Check that the service is running")
def health():
    return {"status": "ok"}

@app.post(
    "/predict",
    response_model=Prediction,
    tags=["Prediction"],
    summary="Classify the emotion in a face image",
    responses={
        400: {"description": "The file is not a valid image."},
        413: {"description": "The image is larger than 5 MB."},
    },
)
async def predict(file: UploadFile = File(description="A face image (.jpg or .png).")):
    """Resize the image, run it through the CNN, and return the most likely
    emotion along with the probability of every class."""
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 5 MB or smaller.")

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Upload a valid image file (.jpg or .png).")

    img = img.resize((IMG_WIDTH, IMG_HEIGHT))                       # PIL takes (width, height)
    x = np.expand_dims(np.array(img, dtype="float32") / 255.0, 0)   # same scaling as training

    probs = model.predict(x, verbose=0)[0]
    best = int(np.argmax(probs))

    return Prediction(
        label=CLASS_CODES[best],
        emotion=EMOTIONS[CLASS_CODES[best]],
        confidence=round(float(probs[best]), 4),
        probabilities={c: round(float(p), 4) for c, p in zip(CLASS_CODES, probs)},
    )