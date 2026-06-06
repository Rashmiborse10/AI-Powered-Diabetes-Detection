from __future__ import annotations

import re
from typing import Dict, List

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    _NLTK_AVAILABLE = True
except Exception:
    nltk = None
    stopwords = None
    word_tokenize = None
    _NLTK_AVAILABLE = False

HEALTH_KNOWLEDGE_TEXT = """
Diabetes is a chronic condition where the body struggles to regulate blood glucose.

Common warning signs include increased thirst, frequent urination, fatigue, blurred vision,
slow healing, and unexplained weight changes.

Type 2 diabetes risk is strongly linked with higher glucose, elevated BMI, insulin resistance,
reduced physical activity, older age, family history, and gestational diabetes history.

Prevention and management usually include balanced meals, weight control, regular exercise,
sleep, stress reduction, hydration, and routine screening such as fasting glucose and HbA1c.

If symptoms are severe, blood sugar is very high, or the person feels unwell, urgent medical
evaluation is recommended.
""".strip()


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    if not _NLTK_AVAILABLE:
        return re.sub(r"\s+", " ", lowered)

    try:
        tokens = word_tokenize(lowered)
        stop_words = set(stopwords.words("english"))
        filtered = [token for token in tokens if token.isalnum() and token not in stop_words]
        return " ".join(filtered)
    except Exception:
        return re.sub(r"\s+", " ", lowered)


def get_risk_guidance(probability: float, is_diabetes: bool, inputs: Dict | None = None) -> Dict[str, List[str] | str]:
    inputs = inputs or {}
    glucose = _safe_float(inputs.get("Glucose"))
    bmi = _safe_float(inputs.get("BMI"))
    age = _safe_float(inputs.get("Age"))
    insulin = _safe_float(inputs.get("Insulin"))

    causes: List[str] = []
    if glucose >= 126:
        causes.append("Elevated glucose levels suggest impaired blood sugar control.")
    if bmi >= 25:
        causes.append("BMI in the overweight or obese range can increase insulin resistance.")
    if insulin >= 100:
        causes.append("Higher insulin values may reflect insulin resistance.")
    if age >= 45:
        causes.append("Risk increases with age, especially after 45.")
    if not causes:
        causes.append("The provided inputs still suggest monitoring because several diabetes risk factors can overlap.")

    diet = [
        "Prefer vegetables, legumes, whole grains, and other high-fiber foods.",
        "Reduce sugary drinks, sweets, and heavily processed foods.",
        "Use lean proteins and healthy fats in controlled portions.",
    ]

    doctor_visit = [
        "Book a medical review for fasting glucose and HbA1c testing.",
        "Seek urgent care if there is vomiting, confusion, chest pain, breathing difficulty, or very high blood sugar.",
        "Discuss a personalized diabetes prevention or management plan with a clinician.",
    ]

    cure = [
        "There is no instant cure, but early screening and treatment can prevent complications.",
        "Type 2 diabetes can often be controlled with lifestyle changes and medication when needed.",
        "If diabetes is confirmed, follow the clinician's treatment and monitoring plan closely.",
    ]

    severity = "high" if is_diabetes or probability >= 55 else "moderate" if probability >= 35 else "lower"

    return {
        "severity": severity,
        "summary": (
            f"Predicted diabetes risk is {probability:.2f}%" if probability else "Risk assessment is available."
        ),
        "causes": causes,
        "cure": cure,
        "diet": diet,
        "doctor_visit": doctor_visit,
    }


def guidance_to_markdown(guidance: Dict[str, List[str] | str]) -> str:
    lines = [
        f"**Risk Level:** {guidance.get('severity', 'unknown').title()}",
        f"**Summary:** {guidance.get('summary', '')}",
        "",
        "**Possible Causes**",
    ]
    lines.extend(f"- {item}" for item in guidance.get("causes", []))
    lines.append("")
    lines.append("**What to Do Now**")
    lines.extend(f"- {item}" for item in guidance.get("cure", []))
    lines.append("")
    lines.append("**Diet Suggestions**")
    lines.extend(f"- {item}" for item in guidance.get("diet", []))
    lines.append("")
    lines.append("**Doctor Visit Recommendation**")
    lines.extend(f"- {item}" for item in guidance.get("doctor_visit", []))
    return "\n".join(lines)


def knowledge_corpus() -> List[str]:
    return [
        HEALTH_KNOWLEDGE_TEXT,
        "Prediabetes and diabetes are best confirmed by a clinician using fasting glucose or HbA1c testing.",
        "Lifestyle changes such as diet quality, exercise, sleep, and weight control lower Type 2 diabetes risk.",
        "A diabetes chatbot should answer questions about symptoms, prevention, complications, diet, and screening.",
    ]
