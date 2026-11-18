# ============================================================
# BRAIN TUMOR AI - FASTAPI BACKEND
# AI RADIOLOGY COPILOT
# ============================================================

import os
import uuid
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.predictor import predict_image
from backend.gradcam import generate_gradcam
from backend.rag_api import ask_medical_assistant
from backend.report_generator import generate_report


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "output",
    "uploads"
)

GRADCAM_DIR = os.path.join(
    BASE_DIR,
    "output",
    "gradcam"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

os.makedirs(
    GRADCAM_DIR,
    exist_ok=True
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Radiology Copilot",
    description=(
        "Brain Tumor Detection API with "
        "Explainable AI using Grad-CAM and "
        "Evidence-Grounded RAG"
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# SERVE GRAD-CAM IMAGES
# ============================================================

app.mount(
    "/gradcam",
    StaticFiles(directory=GRADCAM_DIR),
    name="gradcam"
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "project": "AI Radiology Copilot",
        "model": "BrainTumor_ResNet50_GliomaMeningioma",
        "classes": [
            "glioma",
            "meningioma",
            "notumor",
            "pituitary"
        ]
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate extension
    # --------------------------------------------------------

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    original_filename = file.filename or ""

    extension = os.path.splitext(
        original_filename
    )[1].lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid file type. "
                "Please upload JPG, JPEG or PNG."
            )
        )

    # --------------------------------------------------------
    # Create unique filename
    # --------------------------------------------------------

    file_id = uuid.uuid4().hex

    saved_filename = (
        f"{file_id}{extension}"
    )

    image_path = os.path.join(
        UPLOAD_DIR,
        saved_filename
    )

    # --------------------------------------------------------
    # Save image
    # --------------------------------------------------------

    try:

        with open(
            image_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not save image: {str(e)}"
            )
        )

    # --------------------------------------------------------
    # Run prediction
    # --------------------------------------------------------

    try:

        prediction_result = predict_image(
            image_path
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction failed: {str(e)}"
            )
        )

    # --------------------------------------------------------
    # Generate Grad-CAM
    # --------------------------------------------------------

    gradcam_filename = (
        f"{file_id}_gradcam.jpg"
    )

    try:

        generate_gradcam(
            image_path,
            gradcam_filename
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Grad-CAM failed: {str(e)}"
            )
        )

    # --------------------------------------------------------
    # Return prediction
    # --------------------------------------------------------

    return {
        "success": True,

        "filename": original_filename,

        "prediction": {
            "class": prediction_result[
                "predicted_class"
            ],

            "confidence": round(
                prediction_result[
                    "confidence"
                ],
                2
            ),

            "probabilities": {
                key: round(
                    value,
                    2
                )
                for key, value in
                prediction_result[
                    "probabilities"
                ].items()
            }
        },

        "gradcam": {
            "filename": gradcam_filename,

            "url": (
                f"/gradcam/"
                f"{gradcam_filename}"
            )
        }
    }


# ============================================================
# RAG MEDICAL ASSISTANT
# ============================================================


class RAGQuestion(BaseModel):

    predicted_class: str

    confidence: float

    question: str


# ============================================================
# ASK / RAG ENDPOINT
# ============================================================

@app.post("/ask")
def ask_question(
    data: RAGQuestion
):

    # --------------------------------------------------------
    # Allowed classes
    # --------------------------------------------------------

    allowed_classes = {
        "glioma",
        "meningioma",
        "notumor",
        "pituitary"
    }

    # --------------------------------------------------------
    # Validate class
    # --------------------------------------------------------

    if data.predicted_class not in allowed_classes:

        raise HTTPException(
            status_code=400,
            detail="Invalid predicted class."
        )

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not data.question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # --------------------------------------------------------
    # Generate RAG answer
    # --------------------------------------------------------

    try:

        result = ask_medical_assistant(
            predicted_class=data.predicted_class,
            confidence=data.confidence,
            question=data.question
        )

        # ----------------------------------------------------
        # Make sure result exists
        # ----------------------------------------------------

        if result is None:

            raise HTTPException(
                status_code=500,
                detail=(
                    "RAG assistant returned no result."
                )
            )

        if not isinstance(result, dict):

            raise HTTPException(
                status_code=500,
                detail=(
                    "RAG assistant returned "
                    "an invalid response."
                )
            )

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return {
            "success": True,

            "answer": result.get(
                "answer",
                ""
            ),

            "prediction": result.get(
                "prediction",
                {
                    "class": data.predicted_class,
                    "confidence": data.confidence
                }
            ),

            "sources": result.get(
                "sources",
                []
            )
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            "RAG ASSISTANT ERROR:",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"RAG assistant failed: {str(e)}"
            )
        )


# ============================================================
# AI RADIOLOGY REPORT
# ============================================================


class ReportRequest(BaseModel):

    predicted_class: str

    confidence: float

    probabilities: dict

    gradcam_generated: bool

    rag_answer: str = ""

    sources: list = Field(
        default_factory=list
    )


# ============================================================
# REPORT ENDPOINT
# ============================================================

@app.post("/report")
async def generate_ai_report(
    request: ReportRequest
):

    try:

        # ----------------------------------------------------
        # Validate prediction class
        # ----------------------------------------------------

        valid_classes = {
            "glioma",
            "meningioma",
            "notumor",
            "pituitary"
        }

        predicted_class = (
            request.predicted_class
            .lower()
            .strip()
        )

        if predicted_class not in valid_classes:

            raise HTTPException(
                status_code=400,
                detail="Invalid prediction class."
            )

        # ----------------------------------------------------
        # Generate report
        # ----------------------------------------------------

        report_result = generate_report(
            predicted_class=request.predicted_class,

            confidence=request.confidence,

            probabilities=request.probabilities,

            gradcam_generated=request.gradcam_generated,

            rag_answer=request.rag_answer,

            sources=request.sources
        )

        # ----------------------------------------------------
        # Extract actual report text
        # ----------------------------------------------------

        if isinstance(
            report_result,
            dict
        ):

            report_text = report_result.get(
                "report",
                ""
            )

        else:

            report_text = str(
                report_result
            )

        # ----------------------------------------------------
        # Make sure report contains text
        # ----------------------------------------------------

        if not report_text.strip():

            raise HTTPException(
                status_code=500,
                detail=(
                    "AI report was generated "
                    "but contains no text."
                )
            )

        # ----------------------------------------------------
        # Return clean response
        # ----------------------------------------------------

        return {
            "success": True,

            "report": report_text,

            "prediction": {
                "class": request.predicted_class,

                "confidence": request.confidence,

                "probabilities": request.probabilities
            }
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            "REPORT GENERATION ERROR:",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Report generation failed: {str(e)}"
            )
        )


# ============================================================
# END OF BACKEND
# ============================================================