"""
metrics_page.py — IT Helpdesk Auto-Router: Model Performance Dashboard
=======================================================================
Renders a rich metrics page inside the Streamlit multi-page app.
Loads all pre-saved results CSVs and result images from /results/.
"""

import sys
from pathlib import Path

APP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np

RESULTS = PROJECT_ROOT / "results"

# ─────────────────────────────────────────────────────────────
#  Custom CSS for premium look
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        border: 1px solid #3a3a5c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 4px;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; margin: 6px 0; }
    .metric-label { font-size: 0.85rem; color: #9999bb; text-transform: uppercase; letter-spacing: 1px; }
    .metric-sub   { font-size: 0.78rem; color: #666688; margin-top: 4px; }
    .green  { color: #4ade80; }
    .yellow { color: #facc15; }
    .red    { color: #f87171; }
    .blue   { color: #60a5fa; }
    .section-header {
        font-size: 1.4rem; font-weight: 700;
        border-left: 4px solid #7c3aed;
        padding-left: 12px; margin: 24px 0 16px 0;
        color: #e2e8f0;
    }
    .insight-box {
        background: #1a1a2e; border-left: 3px solid #7c3aed;
        border-radius: 6px; padding: 14px 18px; margin: 10px 0;
        font-size: 0.92rem; color: #c8c8e8;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────────────────
st.title("Model Performance Dashboard")
st.markdown(
    "Real test-set metrics evaluated on a **15% held-out test set (600 tickets)** — "
    "never seen by any model during training or hyperparameter selection."
)
st.divider()

# ─────────────────────────────────────────────────────────────
#  SECTION 1 — TOP-LEVEL KPI SUMMARY
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Project-Level KPIs</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
kpis = [
    ("Category Macro-F1", "1.000", "green",  "LR + TF-IDF", "Best of 6 models"),
    ("Priority Macro-F1", "0.500", "yellow", "GBT + TF-IDF", "Best of 3 models"),
    ("Resolution MAE",    "8.86h", "blue",   "GBT Regressor", "Mean absolute error"),
    ("Resolution R²",     "0.608", "blue",   "GBT Regressor", "Variance explained"),
    ("Test Set Size",     "600",   "green",  "Stratified split", "15% of 5,000 rows"),
]
for col, (label, val, colour, model, sub) in zip([col1,col2,col3,col4,col5], kpis):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {colour}">{val}</div>
        <div class="metric-sub">{model}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")  # spacer

# ─────────────────────────────────────────────────────────────
#  SECTION 2 — CATEGORY CLASSIFICATION
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Task 1 — Ticket Category Classification (6 classes)</div>', unsafe_allow_html=True)

cat_df_raw = pd.read_csv(RESULTS / "category_classification_results.csv", index_col=0)
cat_df = cat_df_raw.copy()
cat_df.columns = [c.strip() for c in cat_df.columns]

tab_table, tab_chart, tab_cm = st.tabs(["Model Comparison Table", "Performance Chart", "Confusion Matrices"])

with tab_table:
    st.markdown("All 6 model × feature combinations evaluated on the **same test set**.")
    # Highlight best macro F1
    def _highlight_best(row):
        best_f1 = cat_df["Macro_F1"].max()
        color = "background-color: #2d4a2d; color: #4ade80;" if row["Macro_F1"] == best_f1 else ""
        return [color] * len(row)

    styled = cat_df.style.apply(_highlight_best, axis=1).format({
        "Accuracy": "{:.4f}", "Macro_F1": "{:.4f}", "Weighted_F1": "{:.4f}"
    })
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown('<div class="insight-box">🔍 <b>Key finding:</b> Logistic Regression + TF-IDF achieves perfect 1.00 Macro-F1. '
                'This is expected — the ticket vocabulary is highly category-specific (confirmed in EDA), '
                'so even a linear classifier can perfectly discriminate all 6 categories. '
                'Word2Vec (averaged document vectors) dilutes this discriminative signal, '
                'which is why LR + Word2Vec drops to 0.78 Macro-F1.</div>', unsafe_allow_html=True)

with tab_chart:
    if (RESULTS / "category_model_comparison.png").exists():
        st.image(str(RESULTS / "category_model_comparison.png"),
                 caption="Category model comparison — Macro-F1 across all 6 configurations",
                 use_column_width=True)
    else:
        st.info("Run Notebook 04 to generate the comparison chart.")

with tab_cm:
    st.markdown("Confusion matrices for each model on the test set. A perfect model has values only on the diagonal.")
    cm_files = sorted(RESULTS.glob("cat_*.png"))
    if cm_files:
        cols = st.columns(2)
        for i, cm_path in enumerate(cm_files):
            model_name = cm_path.stem.replace("cat_", "").replace("_cm", "").replace("_", " ")
            cols[i % 2].image(str(cm_path), caption=model_name, use_column_width=True)
    else:
        st.info("No confusion matrix images found. Run Notebook 04 first.")

st.markdown("")

# Per-class detailed metrics
with st.expander("Per-Class Report — Best Model (LR + TF-IDF) on Test Set"):
    per_class = pd.DataFrame({
        "Class":      ["Access/Login", "Database", "Email", "Hardware", "Network", "Software"],
        "Precision":  [1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
        "Recall":     [1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
        "F1-Score":   [1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
        "Support":    [112,   50,   74,   94,  123,  147],
    })
    st.dataframe(per_class, use_container_width=True, hide_index=True)
    st.caption("Support = number of test samples per class. All classes perfectly classified.")

st.divider()

# ─────────────────────────────────────────────────────────────
#  SECTION 3 — PRIORITY CLASSIFICATION
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Task 2 — Ticket Priority Classification (4 classes)</div>', unsafe_allow_html=True)

pri_df_raw = pd.read_csv(RESULTS / "priority_classification_results.csv", index_col=0)
pri_df = pri_df_raw.copy()
pri_df.columns = [c.strip() for c in pri_df.columns]

tab_p1, tab_p2, tab_p3 = st.tabs(["Model Comparison", "Performance Chart", "Confusion Matrices"])

with tab_p1:
    def _highlight_pri(row):
        best = pri_df["Macro_F1"].max()
        color = "background-color: #2d4a2d; color: #4ade80;" if row["Macro_F1"] == best else ""
        return [color] * len(row)

    styled_p = pri_df.style.apply(_highlight_pri, axis=1).format({
        "Accuracy": "{:.4f}", "Macro_F1": "{:.4f}", "Weighted_F1": "{:.4f}"
    })
    st.dataframe(styled_p, use_container_width=True, hide_index=True)

    st.markdown('<div class="insight-box">🔍 <b>Why priority is harder than category:</b> '
                'Category is signalled by vocabulary — "vpn" → Network, "sql" → Database. '
                'Priority depends on context and urgency cues that users often omit. '
                '"Outlook not working" can be Low (one user) or High (whole team). '
                'The model\'s best Macro-F1 is 0.50 — notably lower than category (1.00). '
                'Critical class (only 48 test samples) has the lowest F1 (0.24) due to class imbalance.</div>',
                unsafe_allow_html=True)

with tab_p2:
    col_pc, col_pd = st.columns(2)
    if (RESULTS / "priority_model_comparison.png").exists():
        col_pc.image(str(RESULTS / "priority_model_comparison.png"),
                     caption="Priority model comparison — Macro-F1", use_column_width=True)
    if (RESULTS / "pri_class_distribution.png").exists():
        col_pd.image(str(RESULTS / "pri_class_distribution.png"),
                     caption="Priority class distribution in training set", use_column_width=True)

with tab_p3:
    pri_cm_files = sorted(RESULTS.glob("pri_*.png"))
    pri_cms = [f for f in pri_cm_files if "_cm" in f.name]
    if pri_cms:
        cols = st.columns(3)
        for i, cm_path in enumerate(pri_cms):
            model_name = cm_path.stem.replace("pri_", "").replace("_cm", "").replace("_", " ")
            cols[i % 3].image(str(cm_path), caption=model_name, use_column_width=True)
    else:
        st.info("No priority confusion matrix images found. Run Notebook 05 first.")

# Per-class detailed metrics (best model: Gradient Boosting)
with st.expander("Per-Class Report — Best Model (Gradient Boosting + TF-IDF) on Test Set"):
    pri_class = pd.DataFrame({
        "Class":     ["Critical", "High", "Low", "Medium"],
        "Precision": [0.29, 0.56, 0.69, 0.58],
        "Recall":    [0.21, 0.53, 0.44, 0.80],
        "F1-Score":  [0.24, 0.54, 0.54, 0.67],
        "Support":   [48, 110, 196, 246],
    })

    def _color_f1(val):
        if isinstance(val, float):
            if val >= 0.65: return "color: #4ade80"
            elif val >= 0.45: return "color: #facc15"
            else: return "color: #f87171"
        return ""

    st.dataframe(
        pri_class.style.applymap(_color_f1, subset=["F1-Score"]).format({
            "Precision": "{:.2f}", "Recall": "{:.2f}", "F1-Score": "{:.2f}"
        }),
        use_container_width=True, hide_index=True
    )
    st.caption("Critical has the lowest F1 (0.24) — it's the rarest class (48 samples). "
               "Medium benefits from highest support (246 samples) and achieves F1=0.67.")

st.divider()

# ─────────────────────────────────────────────────────────────
#  SECTION 4 — REGRESSION
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Task 3 — Resolution Time Regression</div>', unsafe_allow_html=True)

reg_df_raw = pd.read_csv(RESULTS / "regression_results.csv", index_col=0)
reg_df = reg_df_raw.copy()
reg_df.columns = [c.strip() for c in reg_df.columns]

tab_r1, tab_r2, tab_r3 = st.tabs(["Regression Metrics", "Scatter Plots", "Feature Importance"])

with tab_r1:
    # Metric explanation cards
    c1, c2, c3 = st.columns(3)
    c1.markdown("""<div class="metric-card">
        <div class="metric-label">MAE (Best Model)</div>
        <div class="metric-value blue">8.86h</div>
        <div class="metric-sub">Mean Absolute Error</div>
        <div class="metric-sub">Avg prediction off by 8.86 hours</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown("""<div class="metric-card">
        <div class="metric-label">RMSE (Best Model)</div>
        <div class="metric-value yellow">14.12h</div>
        <div class="metric-sub">Root Mean Squared Error</div>
        <div class="metric-sub">Penalises large errors more</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown("""<div class="metric-card">
        <div class="metric-label">R² (Best Model)</div>
        <div class="metric-value blue">0.608</div>
        <div class="metric-sub">Coefficient of Determination</div>
        <div class="metric-sub">61% of variance explained</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("")
    def _highlight_reg(row):
        best_r2 = reg_df["R2"].max()
        color = "background-color: #2d4a2d; color: #4ade80;" if row["R2"] == best_r2 else ""
        return [color] * len(row)

    styled_r = reg_df.style.apply(_highlight_reg, axis=1).format({
        "MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.4f}"
    })
    st.dataframe(styled_r, use_container_width=True, hide_index=True)

    st.markdown('<div class="insight-box">🔍 <b>Interpreting R² = 0.608:</b> '
                'The model explains 61% of the variance in resolution time. '
                'The remaining 39% is due to unobservable factors: individual agent speed, '
                'current ticket backlog, whether parts are available, escalation paths, etc. '
                'Both models perform similarly — Gradient Boosting wins marginally on MAE. '
                'PCA (50 components, ~76% TF-IDF variance retained) ensures text features '
                'don\'t drown out the structured category/priority features.</div>',
                unsafe_allow_html=True)

with tab_r2:
    scatter_files = sorted(RESULTS.glob("reg_*_scatter.png"))
    if scatter_files:
        cols = st.columns(len(scatter_files))
        for i, sc_path in enumerate(scatter_files):
            name = sc_path.stem.replace("reg_", "").replace("_scatter", "").replace("_", " ")
            cols[i].image(str(sc_path), caption=f"{name} — Predicted vs Actual",
                          use_column_width=True)
        st.caption("Points near the red 45° line = accurate predictions. "
                   "Scatter above the line = under-prediction; below = over-prediction.")
    else:
        st.info("No scatter plots found. Run Notebook 06 first.")

    if (RESULTS / "reg_pca_explained_variance.png").exists():
        st.image(str(RESULTS / "reg_pca_explained_variance.png"),
                 caption="PCA: Cumulative explained variance — 50 components retain ~76% of TF-IDF information",
                 use_column_width=True)

with tab_r3:
    if (RESULTS / "reg_feature_importance.png").exists():
        st.image(str(RESULTS / "reg_feature_importance.png"),
                 caption="Feature importance — top 15 features from the best regression model",
                 use_column_width=True)
        st.markdown('<div class="insight-box">🔍 <b>Confirming EDA findings:</b> '
                    'priority_enc and category_enc appear in the top features, '
                    'validating the EDA hypothesis that category and priority are the strongest '
                    'predictors of resolution time. Temporal features (day_of_week, hour_of_day) '
                    'also appear, confirming the Monday volume spike observation.</div>',
                    unsafe_allow_html=True)
    else:
        st.info("Feature importance plot not found. Run Notebook 06 first.")

st.divider()

# ─────────────────────────────────────────────────────────────
#  SECTION 5 — DESIGN DECISIONS TABLE
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Key Engineering Decisions</div>', unsafe_allow_html=True)

decisions = pd.DataFrame([
    ["Class Imbalance",        "class_weight='balanced' / balanced_subsample / compute_sample_weight",
     "Database=8%, Critical=7% — minority class weighting prevents model bias toward majority"],
    ["Feature Engineering",    "TF-IDF (5000 terms, unigrams+bigrams) + PCA (50 components)",
     "EDA showed category-specific vocabulary; PCA prevents TF-IDF from drowning structured features"],
    ["Model Selection",        "Logistic Regression + TF-IDF wins Category; GBT wins Priority",
     "Compared 6 combinations for category, 3 for priority — chose by Macro-F1 on val set"],
    ["Why no LSTM for priority","Already established LSTM underperforms at 5k rows in Notebook 04",
     "Re-running LSTM adds complexity without new insight; LR/RF/GBT suffice"],
    ["Data Leakage Prevention","TF-IDF, PCA, Word2Vec, and LabelEncoders fit on train-only",
     "IDF values computed from train; test projected onto same axes for honest evaluation"],
    ["Routing Engine",         "Pure rule-based lookup tables (TEAM_MAP, SLA_HOURS)",
     "Routing is deterministic policy, not a learning problem; rule-based = auditable + updatable"],
    ["Reproducibility",        "GLOBAL_SEED=42 across numpy, sklearn, PyTorch, gensim",
     "Identical output guaranteed for every re-run"],
], columns=["Decision", "What Was Done", "Why"])

st.dataframe(decisions, use_container_width=True, hide_index=True)

st.divider()

# ─────────────────────────────────────────────────────────────
#  SECTION 6 — EDA FIGURES
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Class Distributions", "Temporal & Resolution", "Text Analysis"])

with eda_tab1:
    if (RESULTS / "eda_01_class_distributions.png").exists():
        st.image(str(RESULTS / "eda_01_class_distributions.png"),
                 caption="Category and priority class distributions — note the imbalance",
                 use_column_width=True)

with eda_tab2:
    col_e1, col_e2 = st.columns(2)
    if (RESULTS / "eda_02_volume_over_time.png").exists():
        col_e1.image(str(RESULTS / "eda_02_volume_over_time.png"),
                     caption="Ticket volume over time — Monday/Tuesday spike", use_column_width=True)
    if (RESULTS / "eda_03_resolution_time_boxplots.png").exists():
        col_e2.image(str(RESULTS / "eda_03_resolution_time_boxplots.png"),
                     caption="Resolution time by category and priority", use_column_width=True)
    if (RESULTS / "eda_06_priority_resolution_correlation.png").exists():
        st.image(str(RESULTS / "eda_06_priority_resolution_correlation.png"),
                 caption="Priority vs resolution time heatmap and violin plot — strong correlation",
                 use_column_width=True)

with eda_tab3:
    if (RESULTS / "eda_05_word_frequencies.png").exists():
        st.image(str(RESULTS / "eda_05_word_frequencies.png"),
                 caption="Top word frequencies per category — near-zero vocabulary overlap confirms TF-IDF advantage",
                 use_column_width=True)

st.divider()
st.caption("📌 All metrics computed on the held-out test set (600 tickets, 15% of 5,000). "
           "No test data was used during model training or hyperparameter selection.")
