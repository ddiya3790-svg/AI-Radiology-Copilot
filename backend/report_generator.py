# ============================================================
# AI RADIOLOGY COPILOT - AI REPORT GENERATOR
# ============================================================

import os
from dotenv import load_dotenv
from groq import Groq


# ------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------

load_dotenv()


# ------------------------------------------------------------
# Groq configuration
# ------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not configured in the .env file."
    )


client = Groq(api_key=GROQ_API_KEY)

MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)


# ------------------------------------------------------------
# Generate AI Report
# ------------------------------------------------------------

def generate_report(
    predicted_class: str,
    confidence: float,
    probabilities: dict,
    gradcam_generated: bool,
    rag_answer: str = "",
    sources: list | None = None
):

    if sources is None:
        sources = []


    # --------------------------------------------------------
    # Format probabilities
    # --------------------------------------------------------

    probability_text = "\n".join(
        [
            f"- {key}: {value:.2f}%"
            for key, value in probabilities.items()
        ]
    )


    # --------------------------------------------------------
    # Format sources
    # --------------------------------------------------------

    source_text = "\n".join(
        [
            f"- {source}"
            if isinstance(source, str)
            else f"- {source.get('source', 'Unknown source')}"
            for source in sources
        ]
    )

    if not source_text:
        source_text = "- No additional sources available."


    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are an AI report-generation component inside an academic
research project called "AI Radiology Copilot".

IMPORTANT SAFETY RULES:

1. This is NOT a medical diagnostic system.
2. Do not claim that the patient definitely has a tumor.
3. Do not invent tumor size, exact location, grade, stage,
   edema, mass effect, enhancement, hormone activity,
   treatment, prognosis, or other clinical findings.
4. Do not interpret Grad-CAM as a tumor segmentation.
5. Clearly distinguish machine-learning prediction from
   medical diagnosis.
6. Use ONLY the information provided below.
7. If information is unavailable, explicitly say that it
   cannot be determined from the available AI output.
8. Do not invent medical facts that are not supported by
   the provided evidence.

------------------------------------------------------------
MODEL OUTPUT
------------------------------------------------------------

Predicted Class:
{predicted_class}

Model Confidence:
{confidence:.2f}%

Class Probabilities:
{probability_text}

Grad-CAM:
{"Generated successfully." if gradcam_generated else "Not available."}

------------------------------------------------------------
RAG EVIDENCE / ASSISTANT INFORMATION
------------------------------------------------------------

{rag_answer if rag_answer else "No additional RAG explanation was provided."}

------------------------------------------------------------
SOURCES
------------------------------------------------------------

{source_text}

------------------------------------------------------------
REPORT FORMAT
------------------------------------------------------------

Generate a concise, professional AI-assisted research report
using exactly these sections:

1. Analysis Summary
2. Model Prediction
3. Confidence and Class Probabilities
4. Explainability
5. Evidence Summary
6. Important Limitations
7. Recommended Next Step
8. Research Disclaimer

For "Model Prediction", explain that the predicted class is
the class assigned by the trained image-classification model.

For "Confidence and Class Probabilities", explain that these
are model probabilities and are NOT clinical probabilities.

For "Explainability", explain that Grad-CAM highlights image
regions that influenced the model's prediction, but it is not
a precise tumor boundary or segmentation.

For "Evidence Summary", summarize only information supported
by the supplied RAG evidence.

For "Recommended Next Step", say that the AI result should be
reviewed by an appropriately qualified healthcare professional
if this were a real clinical situation.

For "Research Disclaimer", clearly state that this is an
academic AI research project and must not be used as a medical
diagnosis or substitute for professional medical advice.

Do not use markdown tables.
Use clear headings and bullet points where appropriate.
"""


    # --------------------------------------------------------
    # Call Groq
    # --------------------------------------------------------

    try:

        response = client.chat.completions.create(

            model=MODEL_NAME,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate safe, evidence-grounded "
                        "AI research reports."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2,

            max_tokens=1800
        )


        report = response.choices[0].message.content.strip()


        return {
            "success": True,
            "report": report,
            "model": MODEL_NAME
        }


    except Exception as e:

        return {
            "success": False,
            "report": "",
            "model": MODEL_NAME,
            "error": str(e)
        }


# ------------------------------------------------------------
# Local test
# ------------------------------------------------------------

if __name__ == "__main__":

    test_result = generate_report(

        predicted_class="pituitary",

        confidence=95.68,

        probabilities={
            "glioma": 0.01,
            "meningioma": 4.30,
            "notumor": 0.00,
            "pituitary": 95.68
        },

        gradcam_generated=True,

        rag_answer=(
            "The AI model classified the image as pituitary. "
            "This is a machine-learning prediction and does "
            "not independently confirm a medical diagnosis."
        ),

        sources=[
            "mri_basics.txt",
            "brain_tumor_types.txt",
            "brain_tumor_basics.txt"
        ]
    )


    print("\n====================================")
    print("AI REPORT GENERATOR TEST")
    print("====================================\n")

    if test_result["success"]:

        print(test_result["report"])

    else:

        print("Report generation failed:")
        print(test_result["error"])

    print("\n====================================")