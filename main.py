"""
Catfish Detector — FastAPI Backend
Serves the /detect_catfish endpoint consumed by MatchOutcomePredictor.html.

Start:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import warnings
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

warnings.filterwarnings("ignore")

# ── Model Loading ────────────────────────────────────────────────────────────

_BASE = Path(__file__).parent

for _required in ("model.onnx", "scaler_mean.npy", "scaler_scale.npy"):
    if not (_BASE / _required).exists():
        raise FileNotFoundError(
            f"Missing {_required}. Run convert_to_onnx.py locally first."
        )

_session      = ort.InferenceSession(str(_BASE / "model.onnx"))
_scaler_mean  = np.load(str(_BASE / "scaler_mean.npy"))
_scaler_scale = np.load(str(_BASE / "scaler_scale.npy"))

# feature_columns order must match what was used during training — read from a
# small sidecar file produced by convert_to_onnx.py, or hardcode here if stable.
_FEATURE_COLUMNS_PATH = _BASE / "feature_columns.npy"
if _FEATURE_COLUMNS_PATH.exists():
    _feature_columns: List[str] = np.load(str(_FEATURE_COLUMNS_PATH), allow_pickle=True).tolist()
else:
    raise FileNotFoundError(
        "Missing feature_columns.npy. Run convert_to_onnx.py locally first."
    )

# ── Encoding Maps ────────────────────────────────────────────────────────────
# LabelEncoder was fit on sorted unique values from the training CSV.

INCOME_ENC = {
    "High": 0, "Low": 1, "Lower-Middle": 2, "Middle": 3,
    "Upper-Middle": 4, "Very High": 5, "Very Low": 6,
}

EDUCATION_ENC = {
    "Associate's": 0, "Bachelor's": 1, "Diploma": 2, "High School": 3,
    "MBA": 4, "Master's": 5, "No Formal Education": 6, "PhD": 7, "Postdoc": 8,
}

# All tags present as model features (tag_<name> columns)
ALL_TAGS = [
    "Gaming", "Traveling", "Photography", "Reading", "Art", "Music",
    "Dancing", "Movies", "Fitness", "Coding", "Binge-Watching", "Spirituality",
    "Motorcycling", "Yoga", "Writing", "Hiking", "Makeup", "Investing",
    "Clubbing", "Skating",
]

# One-hot categories — baseline (dropped) class listed in comment
GENDER_OHE   = ["Genderfluid", "Male", "Non-binary", "Prefer Not to Say", "Transgender"]  # baseline: Female
ORIENT_OHE   = ["Bisexual", "Demisexual", "Gay", "Lesbian", "Pansexual", "Queer", "Straight"]  # baseline: Asexual
LOCATION_OHE = ["Remote Area", "Rural", "Small Town", "Suburban", "Urban"]  # baseline: Metro
TIME_OHE     = ["Afternoon", "Early Morning", "Evening", "Late Night", "Morning"]  # baseline: After Midnight

SUSPICION_MAP = {0: "Minimal", 1: "Low", 2: "Moderate", 3: "Elevated", 4: "High", 5: "Severe", 6: "Critical"}

# ── Pydantic Schema ──────────────────────────────────────────────────────────

class ProfilePayload(BaseModel):
    # Categorical identity fields
    gender:             str
    sexual_orientation: str
    location_type:      str
    income_bracket:     str
    education_level:    str

    # Comma-separated interest tags string
    interest_tags: str = ""

    # Label fields (not fed to model directly, kept for logging)
    app_usage_time_label: str = ""
    swipe_right_label:    str = ""

    # Numeric usage metrics
    app_usage_time_minute: int   = Field(..., ge=0, le=1440)
    swip_right_ratio:      float = Field(..., ge=0.0, le=1.0)
    likes_received:        int   = Field(..., ge=0)
    mutual_matches:        int   = Field(..., ge=0)
    profile_pics_count:    int   = Field(..., ge=0, le=20)
    bio_length:            int   = Field(..., ge=0, le=500)
    message_sent_count:    int   = Field(..., ge=0)
    emoji_usage_rate:      float = Field(..., ge=0.0, le=1.0)
    last_active_hour:      int   = Field(..., ge=0, le=23)

    # Time-of-day (one-hot feature)
    swipe_time_of_day: str = ""

    @field_validator("education_level", mode="before")
    @classmethod
    def normalise_education(cls, v: str) -> str:
        # Normalise curly/smart apostrophes to straight ones
        return v.replace("’", "'").replace("‘", "'").strip()

    @field_validator("income_bracket", "gender", "sexual_orientation",
                     "location_type", "swipe_time_of_day", mode="before")
    @classmethod
    def strip_strings(cls, v: str) -> str:
        return v.strip()


class DetectionResponse(BaseModel):
    catfish_prob:          int
    authenticity_score:    int
    suspicion_index:       str
    behavioral_consistency: int
    risk_level:            str
    red_flags:             List[str]


# ── Feature Engineering ──────────────────────────────────────────────────────

def _build_feature_vector(p: ProfilePayload) -> pd.DataFrame:
    tags = {t.strip() for t in p.interest_tags.split(",") if t.strip()}

    feat: dict = {
        # Raw numerics
        "app_usage_time_min": p.app_usage_time_minute,
        "swipe_right_ratio":  p.swip_right_ratio,
        "likes_received":     p.likes_received,
        "mutual_matches":     p.mutual_matches,
        "profile_pics_count": p.profile_pics_count,
        "bio_length":         p.bio_length,
        "message_sent_count": p.message_sent_count,
        "emoji_usage_rate":   p.emoji_usage_rate,
        "last_active_hour":   p.last_active_hour,

        # Engineered features (same formulae used during training)
        "match_conversion_rate": p.mutual_matches / max(p.likes_received, 1),
        "msg_per_match":         p.message_sent_count / max(p.mutual_matches, 1),
        "profile_completeness":  (
            min(p.profile_pics_count / 6.0, 1.0) * 0.5
            + min(p.bio_length / 300.0, 1.0) * 0.5
        ),
        "usage_intensity": p.app_usage_time_minute / 1440.0,
    }

    # Binary tag flags
    for tag in ALL_TAGS:
        feat[f"tag_{tag}"] = 1 if tag in tags else 0

    # Ordinal-encoded demographics
    feat["income_bracket_enc"]  = INCOME_ENC.get(p.income_bracket, 0)
    feat["education_level_enc"] = EDUCATION_ENC.get(p.education_level, 1)

    # One-hot: gender (Female = baseline)
    for g in GENDER_OHE:
        feat[f"gender_{g}"] = 1 if p.gender == g else 0

    # One-hot: sexual orientation (Asexual = baseline)
    for so in ORIENT_OHE:
        feat[f"sexual_orientation_{so}"] = 1 if p.sexual_orientation == so else 0

    # One-hot: location type (Metro = baseline)
    for lt in LOCATION_OHE:
        feat[f"location_type_{lt}"] = 1 if p.location_type == lt else 0

    # One-hot: swipe time of day (After Midnight = baseline)
    for st in TIME_OHE:
        feat[f"swipe_time_of_day_{st}"] = 1 if p.swipe_time_of_day == st else 0

    return pd.DataFrame([feat])[_feature_columns].astype(np.float32)


def _compute_red_flags(p: ProfilePayload) -> List[str]:
    flags: List[str] = []
    if p.profile_pics_count <= 1:
        flags.append("low_pics")
    if p.bio_length < 20:
        flags.append("empty_bio")
    if p.swip_right_ratio > 0.9:
        flags.append("mass_swiper")
    if p.mutual_matches / max(p.likes_received, 1) < 0.05:
        flags.append("low_match_rate")
    if 1 <= p.last_active_hour <= 5:
        flags.append("odd_hours")
    if p.message_sent_count < 5 and p.mutual_matches > 10:
        flags.append("no_messages")
    return flags


def _derive_scores(catfish_prob: int, red_flags: List[str], tags: str) -> dict:
    tag_count   = len([t for t in tags.split(",") if t.strip()])
    auth_score  = max(5, 100 - catfish_prob - (8 if tag_count < 2 else 0))
    consistency = max(5, 100 - len(red_flags) * 16)
    suspicion   = SUSPICION_MAP[min(len(red_flags), 6)]
    risk_level  = (
        "Critical" if catfish_prob >= 70 else
        "High"     if catfish_prob >= 50 else
        "Medium"   if catfish_prob >= 30 else
        "Low"
    )
    return {
        "authenticity_score":    int(auth_score),
        "behavioral_consistency": int(consistency),
        "suspicion_index":       suspicion,
        "risk_level":            risk_level,
    }


# ── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Catfish Detector API",
    description="LightGBM-powered catfish profile detection endpoint.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your frontend origin in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": "lightgbm-onnx"}


@app.post("/detect_catfish", response_model=DetectionResponse)
def detect_catfish(payload: ProfilePayload) -> DetectionResponse:
    try:
        X = _build_feature_vector(payload)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Feature engineering failed: {exc}") from exc

    try:
        X_arr    = X.values  # already float32 from _build_feature_vector
        X_scaled = ((X_arr - _scaler_mean) / _scaler_scale).astype(np.float32)
        input_name = _session.get_inputs()[0].name
        outputs    = _session.run(None, {input_name: X_scaled})
        # outputs[1] is a list of dicts {class_label: probability}
        catfish_prob_raw: float = float(outputs[1][0][1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

    catfish_prob = int(round(catfish_prob_raw * 100))
    red_flags    = _compute_red_flags(payload)
    scores       = _derive_scores(catfish_prob, red_flags, payload.interest_tags)

    return DetectionResponse(
        catfish_prob=catfish_prob,
        red_flags=red_flags,
        **scores,
    )


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
