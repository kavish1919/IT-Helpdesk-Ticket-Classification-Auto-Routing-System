"""
routing_engine.py  —  IT Helpdesk Ticket Classifier
=====================================================
Provides the route_ticket() function that runs a raw ticket description
through the full ML pipeline and returns a routing + SLA-breach decision.
"""

import numpy as np
import joblib
import datetime
from pathlib import Path

from src.preprocessing import clean_text

# ── paths ──────────────────────────────────────────────────────────────────────
_ROOT   = Path(__file__).resolve().parent.parent
_MODELS = _ROOT / 'models'

# ── business logic lookup tables ───────────────────────────────────────────────
# These tables encode domain knowledge, not ML predictions.
# They should be maintained by the ITSM team and updated as org structure changes.

TEAM_MAP = {
    "Network":      "Infrastructure Team",
    "Hardware":     "Hardware Support Team",
    "Software":     "Application Support Team",
    "Access/Login": "Identity & Access Management Team",
    "Email":        "Messaging & Collaboration Team",
    "Database":     "Database Administration Team",
}

SLA_HOURS = {
    "Critical": 4,    # 4-hour SLA — immediate escalation
    "High":     8,    # 8-hour SLA — same business day
    "Medium":   24,   # 24-hour SLA — next business day
    "Low":      72,   # 72-hour SLA — 3 business days
}

# ── global model registry (populated by load_models()) ────────────────────────
_models = {}

def load_models(models_dir: str = None) -> None:
    """
    Load all saved model artifacts into the global registry.
    Call this ONCE at application startup (not on every request).

    Parameters
    ----------
    models_dir : str, optional
        Path to the models/ directory. Defaults to PROJECT_ROOT/models.
    """
    global _models
    mdir = Path(models_dir) if models_dir else _MODELS

    _models['tfidf']       = joblib.load(mdir / 'tfidf_vectorizer.joblib')
    _models['pca']         = joblib.load(mdir / 'pca_tfidf.joblib')
    _models['le_cat']      = joblib.load(mdir / 'label_encoder_category.joblib')
    _models['le_pri']      = joblib.load(mdir / 'label_encoder_priority.joblib')
    _models['cat_model']   = joblib.load(mdir / 'best_category_model.joblib')
    _models['pri_model']   = joblib.load(mdir / 'best_priority_model.joblib')
    _models['reg_model']   = joblib.load(mdir / 'best_resolution_model.joblib')

    print('Routing engine: all models loaded successfully.')
    print(f'  Category model : {type(_models["cat_model"]).__name__}')
    print(f'  Priority model : {type(_models["pri_model"]).__name__}')
    print(f'  Regression model: {type(_models["reg_model"]).__name__}')


def _require_models():
    """Raise a clear error if load_models() has not been called."""
    if not _models:
        raise RuntimeError(
            "Models not loaded. Call load_models() before route_ticket()."
        )


# ── internal: build regression feature vector ─────────────────────────────────
def _build_regression_features(cleaned_text: str,
                                cat_enc: int,
                                pri_enc: int,
                                created_at: datetime.datetime = None) -> np.ndarray:
    """
    Build the 54-feature vector used by the resolution time regressor.

    Feature layout (matches Notebook 06):
      [0] category_enc
      [1] priority_enc
      [2] day_of_week   (0=Monday)
      [3] hour_of_day   (0-23)
      [4..53] PCA components of TF-IDF

    If created_at is None, the current time is used for temporal features.
    """
    if created_at is None:
        created_at = datetime.datetime.now()

    # Structured features
    dow  = created_at.weekday()   # 0=Monday, 6=Sunday
    hour = created_at.hour
    struct = np.array([[cat_enc, pri_enc, dow, hour]], dtype=float)

    # Text features: TF-IDF → PCA
    tfidf_vec = _models['tfidf'].transform([cleaned_text])
    pca_vec   = _models['pca'].transform(tfidf_vec.toarray())   # shape (1, 50)

    # Concatenate
    return np.hstack([struct, pca_vec])   # shape (1, 54)


# ── public API ─────────────────────────────────────────────────────────────────
def route_ticket(ticket_text: str,
                 created_at: datetime.datetime = None) -> dict:
    """
    Full pipeline: raw ticket text  →  routing + SLA decision.

    Steps
    -----
    1. Preprocess text (lowercase, remove punctuation, stopwords, stemming)
    2. Predict category  (best model from Notebook 04)
    3. Predict priority  (best model from Notebook 05)
    4. Predict expected resolution time (regressor from Notebook 06)
    5. Apply business rules → assigned_team, sla_target, sla_breach_risk

    Parameters
    ----------
    ticket_text : str            — raw free-text ticket description
    created_at  : datetime, opt  — ticket creation time (defaults to now)

    Returns
    -------
    dict with the following keys:
      ticket_text                 : original (truncated to 120 chars for display)
      category                   : predicted category
      priority                   : predicted priority
      predicted_resolution_hours : float (hours)
      assigned_team              : team name string
      sla_target_hours           : int
      sla_breach_risk            : bool
      confidence_note            : str
    """
    _require_models()

    # ── Step 1: preprocess ────────────────────────────────────────────────────
    cleaned = clean_text(ticket_text)
    tfidf_vec = _models['tfidf'].transform([cleaned])

    # ── Step 2: predict category ──────────────────────────────────────────────
    cat_enc  = int(_models['cat_model'].predict(tfidf_vec)[0])
    category = _models['le_cat'].inverse_transform([cat_enc])[0]

    # ── Step 3: predict priority ──────────────────────────────────────────────
    pri_enc  = int(_models['pri_model'].predict(tfidf_vec)[0])
    priority = _models['le_pri'].inverse_transform([pri_enc])[0]

    # ── Step 4: predict resolution time ──────────────────────────────────────
    reg_features  = _build_regression_features(cleaned, cat_enc, pri_enc, created_at)
    pred_hours    = float(_models['reg_model'].predict(reg_features)[0])
    pred_hours    = round(max(0.5, pred_hours), 1)   # floor at 30 min

    # ── Step 5: business rules ────────────────────────────────────────────────
    assigned_team    = TEAM_MAP.get(category, "General IT Support")
    sla_target       = SLA_HOURS.get(priority, 24)
    sla_breach_risk  = bool(pred_hours > sla_target)

    return {
        "ticket_text":                ticket_text[:120] + ("..." if len(ticket_text) > 120 else ""),
        "category":                   category,
        "priority":                   priority,
        "predicted_resolution_hours": pred_hours,
        "assigned_team":              assigned_team,
        "sla_target_hours":           sla_target,
        "sla_breach_risk":            sla_breach_risk,
        "confidence_note":            "Predictions are probabilistic. Human review recommended for Critical tickets.",
    }


def batch_route(ticket_texts: list,
                created_ats: list = None) -> list:
    """
    Route a list of tickets in batch.
    Returns a list of dicts (same format as route_ticket).
    """
    if created_ats is None:
        created_ats = [None] * len(ticket_texts)
    return [route_ticket(t, ca) for t, ca in zip(ticket_texts, created_ats)]
