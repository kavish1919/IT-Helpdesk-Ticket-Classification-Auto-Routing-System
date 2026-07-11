"""
streamlit_app.py  —  IT Helpdesk Ticket Auto-Router Demo
=========================================================
A single-page Streamlit app that exposes the routing engine via a web UI.

RUN WITH:
  cd it-ticket-classifier
  streamlit run app/streamlit_app.py

REQUIRES:
  All notebooks 01-06 must have been run first so that model artifacts
  exist in the models/ directory.
"""

import sys
from pathlib import Path

# ── project root on the path so src/ imports work ─────────────────────────────
APP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import time

from src.routing_engine import load_models, route_ticket, TEAM_MAP, SLA_HOURS

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IT Helpdesk Auto-Router",
    layout="centered",
)

# ── load models once (cached so they survive re-runs) ─────────────────────────
@st.cache_resource(show_spinner="Loading ML models...")
def _load():
    load_models(models_dir=str(PROJECT_ROOT / "models"))
    return True

models_ready = _load()

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — navigation + project info
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("IT Auto-Router")
    page = st.radio(
        "Navigate",
        ["Ticket Router", "Model Metrics"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("""
**IT Helpdesk Ticket Classification & Auto-Routing System**

A portfolio ML project demonstrating:
- Text classification (6 categories)
- Priority prediction (4 levels)
- Resolution time regression
- SLA breach risk flagging

**Tech stack:**
- Python · NLTK · TF-IDF
- scikit-learn · gensim
- PyTorch (LSTM)
- Streamlit (this app)

---
**SLA Targets**
| Priority | Target |
|---|---|
| Critical | 4 hours |
| High | 8 hours |
| Medium | 24 hours |
| Low | 72 hours |

---
*Dataset: synthetic (5 000 rows)*  
*All models trained on historical tickets.*
""")

# ── Route to the correct page ─────────────────────────────────────────────────
if page == "Model Metrics":
    import importlib, sys
    metrics_path = str(APP_DIR)
    if metrics_path not in sys.path:
        sys.path.insert(0, metrics_path)
    # Execute metrics_page as a module in the current Streamlit context
    exec(open(APP_DIR / "metrics_page.py", encoding="utf-8").read())
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
st.title("IT Helpdesk Auto-Router")
st.markdown(
    "Enter a support ticket description below. "
    "The system will predict its **category**, **priority**, "
    "**expected resolution time**, and flag **SLA breach risk** — "
    "then route it to the correct team automatically."
)

st.divider()

# ── callback for quick-fill buttons ──────────────────────────────────────────
def _set_example_text(sample_text: str):
    st.session_state["ticket_input"] = sample_text

# ── ticket input ───────────────────────────────────────────────────────────────
ticket_input = st.text_area(
    label="**Ticket Description**",
    height=140,
    placeholder=(
        "Examples:\n"
        "• vpn not connecting since morning, getting error 401\n"
        "• outlook crashes when opening attachments, urgent\n"
        "• locked out of account after 5 failed attempts\n"
        "• production database PROD_DB down, all transactions failing"
    ),
    key="ticket_input",
)

# ── example tickets (quick-fill buttons) ─────────────────────────────────────
st.caption("Or try an example:")
col1, col2, col3 = st.columns(3)
examples = {
    "Network"  : "vpn not connecting since morning, error 401, cannot work at all",
    "Software" : "outlook keeps crashing when i open attachments since the update",
    "Database" : "production database PROD_DB is down, all transactions failing urgent",
}
for col, (label, text) in zip([col1, col2, col3], examples.items()):
    col.button(label, key=f"ex_{label}", on_click=_set_example_text, args=(text,))

st.divider()

# ── analyse button ─────────────────────────────────────────────────────────────
analyse = st.button("Analyse Ticket", type="primary", use_container_width=True)

if analyse:
    text = st.session_state.get("ticket_input", "").strip()

    if not text:
        st.warning("Please enter a ticket description before clicking Analyse.")
    else:
        with st.spinner("Running ML pipeline..."):
            time.sleep(0.3)   # small delay for UX — feels more "real"
            result = route_ticket(text)

        # ── Results ────────────────────────────────────────────────────────────
        st.subheader("Routing Decision")

        # Top metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric(
            label="Category",
            value=result["category"],
        )
        m2.metric(
            label="Priority",
            value=result["priority"],
        )
        m3.metric(
            label="Est. Resolution",
            value=f"{result['predicted_resolution_hours']}h",
            delta=f"SLA: {result['sla_target_hours']}h",
            delta_color="inverse",
        )

        st.divider()

        # Routing details
        col_a, col_b = st.columns(2)
        col_a.info(f"**Assigned Team**\n\n{result['assigned_team']}")
        col_b.info(f"**SLA Target**\n\n{result['sla_target_hours']} hours")

        # SLA breach risk flag
        if result["sla_breach_risk"]:
            st.error(
                "**SLA Breach Risk Detected**  \n"
                f"Predicted resolution time ({result['predicted_resolution_hours']}h) "
                f"exceeds the {result['priority']} SLA target of {result['sla_target_hours']}h.  \n"
                "**Action:** Escalate immediately to team lead."
            )
        else:
            st.success(
                f"**Within SLA Target**  \n"
                f"Predicted resolution ({result['predicted_resolution_hours']}h) "
                f"is within the {result['sla_target_hours']}h {result['priority']} SLA."
            )

        st.divider()

        # Raw JSON output (expandable)
        with st.expander("View raw prediction output"):
            st.json(result)

        # Confidence note
        st.caption(result['confidence_note'])

# ── batch mode ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Batch Mode — Upload Tickets CSV")
st.markdown(
    "Upload a CSV with a `ticket_text` column to route multiple tickets at once."
)

uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")
if uploaded is not None:
    try:
        batch_df = pd.read_csv(uploaded)
        if "ticket_text" not in batch_df.columns:
            st.error("CSV must contain a column named `ticket_text`.")
        else:
            with st.spinner(f"Routing {len(batch_df)} tickets..."):
                from src.routing_engine import batch_route
                results_list = batch_route(batch_df["ticket_text"].tolist())
                out_df = pd.DataFrame(results_list).drop(columns=["confidence_note"])

            st.success(f"Routed {len(out_df)} tickets successfully.")
            st.dataframe(out_df, use_container_width=True)

            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Results CSV",
                data=csv_bytes,
                file_name="routing_results.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Error processing file: {e}")
