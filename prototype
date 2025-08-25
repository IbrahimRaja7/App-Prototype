import io
import textwrap
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# ---------------------------
# App meta
# ---------------------------
APP_TITLE = "UN SDG 3 Health Navigator"
APP_SUBTITLE = "Screening & Education for Child Malnutrition, Pregnancy Loss Risk, and Older Adult Health"
VERSION = "1.0.0"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🩺",
    layout="wide",
)

# ---------------------------
# Utilities
# ---------------------------

def ribbon(text: str, emoji: str = "🎯"):
    st.markdown(
        f"""
        <div style='background: linear-gradient(90deg, #e0f2fe, #f0fdf4); padding: 0.75rem 1rem; border-radius: 0.75rem; border: 1px solid #e5e7eb; font-weight: 600;'>
            {emoji} {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def disclaimer_box():
    st.markdown(
        """
        > **Educational screening only — not medical advice.** This app supports **UN SDG 3 (Good Health and Well‑Being)** by helping teams perform *basic screening and data review*. 
        > It **does not** diagnose conditions and **must not** replace professional clinical judgment. If risk is indicated, seek appropriate medical evaluation.
        """
    )


# ---------------------------
# Child malnutrition module
# ---------------------------

@st.cache_data
def muac_interpretation(muac_cm: Optional[float], age_months: Optional[int]) -> dict:
    """Interpret MUAC for ages 6–59 months per common screening cut‑offs.
    Returns a dict with category and guidance.
    """
    if muac_cm is None or np.isnan(muac_cm) or age_months is None:
        return {"category": "—", "severity": 0, "note": "Enter MUAC and age."}

    if age_months < 6 or age_months > 59:
        return {
            "category": "Out of standard MUAC range",
            "severity": 0,
            "note": "MUAC 6–59 months only. Consider weight‑for‑height or local protocols.",
        }

    if muac_cm < 11.5:
        return {
            "category": "Severe Acute Malnutrition (SAM)",
            "severity": 3,
            "note": "Urgent referral per local IMAM/CMAM guidelines.",
        }
    elif 11.5 <= muac_cm < 12.5:
        return {
            "category": "Moderate Acute Malnutrition (MAM)",
            "severity": 2,
            "note": "Supplementary feeding & follow‑up as per protocol.",
        }
    else:
        return {
            "category": "No acute malnutrition by MUAC",
            "severity": 1,
            "note": "Continue routine growth monitoring and nutrition counseling.",
        }


def child_malnutrition_ui():
    st.header("Child Malnutrition — Screening (6–59 months)")
    st.caption("Focus area of **UN SDG 3.2**: end preventable deaths of newborns and children under five.")
    disclaimer_box()

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        age_months = st.number_input("Age (months)", min_value=0, max_value=179, step=1)
    with col2:
        sex = st.selectbox("Sex", ["Female", "Male", "Intersex/Other", "Prefer not to say"])
    with col3:
        weight = st.number_input("Weight (kg)", min_value=0.0, step=0.1, format="%.1f")
    with col4:
        height = st.number_input("Height/Length (cm)", min_value=0.0, step=0.1, format="%.1f")

    col5, col6, col7 = st.columns([1, 1, 1])
    with col5:
        muac = st.number_input("MUAC (cm)", min_value=0.0, step=0.1, format="%.1f")
    with col6:
        edema = st.selectbox("Bilateral pitting edema?", ["No", "Yes"]) 
    with col7:
        breastfeeding = st.selectbox("Exclusive breastfeeding (0–6 mo)?", ["N/A", "Yes", "No"]) 

    # Calculations
    bmi = (weight / ((height / 100) ** 2)) if weight and height else np.nan
    muac_result = muac_interpretation(muac if muac else np.nan, age_months if age_months else None)

    st.subheader("Results")
    mcol1, mcol2, mcol3 = st.columns([1, 1, 2])
    with mcol1:
        st.metric("BMI (kg/m²)", f"{bmi:.1f}" if not np.isnan(bmi) else "—")
    with mcol2:
        st.metric("MUAC Category", muac_result["category"])
    with mcol3:
        ribbon(muac_result["note"], emoji="📌")

    if edema == "Yes":
        st.warning("Presence of bilateral pitting edema suggests **severe acute malnutrition** — follow urgent referral pathways.")

    st.divider()

    st.markdown("**Counseling prompts** (adapt to local guidelines):")
    st.markdown(
        """
        - Promote **exclusive breastfeeding** for the first 6 months; timely, adequate **complementary feeding** thereafter.
        - Encourage **diverse diets**: grains/tubers; legumes/nuts; dairy; flesh foods; eggs; vitamin‑A rich fruits/vegetables; other fruits/vegetables.
        - Check **immunization status**, vitamin A, deworming where appropriate.
        - Address **WASH**: safe water, sanitation, and hygiene to reduce infections.
        - Screen for **illness**, feeding difficulties, and **household food insecurity**; refer to social support.
        """
    )

    # Optional data capture
    with st.expander("Save screening record"):
        caregiver_id = st.text_input("Caregiver/Child ID (optional)")
        notes = st.text_area("Notes")
        if st.button("Add to session log"):
            rec = {
                "timestamp": datetime.utcnow().isoformat(),
                "module": "child_malnutrition",
                "age_months": age_months,
                "sex": sex,
                "weight_kg": weight,
                "height_cm": height,
                "bmi": None if np.isnan(bmi) else round(bmi, 1),
                "muac_cm": muac,
                "muac_category": muac_result["category"],
                "edema": edema,
                "breastfeeding": breastfeeding,
                "id": caregiver_id,
                "notes": notes,
            }
            st.session_state.setdefault("_session_log", []).append(rec)
            st.success("Record added to session log.")


# ---------------------------
# Pregnancy loss / miscarriage risk education
# ---------------------------

def miscarriage_risk_score(age: int, prior_losses: int, bmi: float, smoker: bool, alcohol: bool, chronic: bool, infection: bool, occupational: bool) -> dict:
    """Very simple additive educational score (NOT clinical). Returns label and guidance."""
    score = 0
    # Age factors
    if age >= 40:
        score += 3
    elif age >= 35:
        score += 2
    elif age < 18:
        score += 1

    # History
    if prior_losses >= 3:
        score += 3
    elif prior_losses == 2:
        score += 2
    elif prior_losses == 1:
        score += 1

    # Lifestyle / medical
    if bmi >= 30 or bmi < 18.5:
        score += 1
    if smoker:
        score += 1
    if alcohol:
        score += 1
    if chronic:  # e.g., thyroid disease, uncontrolled diabetes, antiphospholipid syndrome, uterine anomalies (self-reported)
        score += 2
    if infection:
        score += 1
    if occupational:  # heavy metals/solvents/radiation/very heavy physical labor (self-reported)
        score += 1

    if score >= 7:
        label = "Higher‑than‑average risk indicators"
        note = "Recommend timely prenatal care and evaluation for recurrent pregnancy loss where applicable."
    elif score >= 4:
        label = "Some elevated risk indicators"
        note = "Address modifiable factors (smoking, alcohol, weight) and seek early antenatal assessment."
    else:
        label = "No major risk indicators flagged here"
        note = "Maintain healthy habits and routine prenatal care when pregnant."

    return {"score": score, "label": label, "note": note}


def miscarriage_ui():
    st.header("Pregnancy Loss (Miscarriage) — Risk Education")
    st.caption("Supports **UN SDG 3.1**: reduce global maternal mortality and improve reproductive health services.")
    disclaimer_box()

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age (years)", min_value=10, max_value=60, value=28, step=1)
        prior_losses = st.number_input("Prior pregnancy losses (confirmed)", min_value=0, max_value=10, value=0, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0, step=0.1)
    with col2:
        smoker = st.checkbox("Currently smoking or vaping")
        alcohol = st.checkbox("Regular alcohol use")
        chronic = st.checkbox("Chronic conditions or uterine anomalies (self‑reported)")
    with col3:
        infection = st.checkbox("Recent significant infection (e.g., high fever)")
        occupational = st.checkbox("Occupational/environmental hazards (solvents, heavy metals, radiation, heavy labor)")
        preconception = st.checkbox("Preconception planning visit completed")

    result = miscarriage_risk_score(age, prior_losses, bmi, smoker, alcohol, chronic, infection, occupational)

    st.subheader("Summary")
    sc1, sc2 = st.columns([1, 3])
    with sc1:
        st.metric("Screening score", result["score"])
    with sc2:
        ribbon(f"{result['label']} — {result['note']}", emoji="🧭")

    st.markdown("**Health prompts** (individualize to local guidelines):")
    st.markdown(
        """
        - **Before pregnancy:** optimize **chronic conditions** (e.g., diabetes, thyroid), review medications, update vaccines, and consider **folic acid** per guidelines.
        - Avoid **tobacco** and **alcohol**; discuss weight management to reach a healthy BMI.
        - Seek early **antenatal care** with any positive test; report bleeding, severe pain, or fever promptly.
        - For recurrent losses, request evaluation (anatomy, endocrine, genetic, autoimmune, and thrombophilia as clinically indicated).
        """
    )

    with st.expander("Save screening record"):
        person_id = st.text_input("Client ID (optional)")
        notes = st.text_area("Notes")
        if st.button("Add to session log", key="miscarriage_add"):
            rec = {
                "timestamp": datetime.utcnow().isoformat(),
                "module": "pregnancy_loss",
                "age": age,
                "prior_losses": prior_losses,
                "bmi": bmi,
                "smoker": smoker,
                "alcohol": alcohol,
                "chronic": chronic,
                "infection": infection,
                "occupational": occupational,
                "preconception": preconception,
                "id": person_id,
                "notes": notes,
            }
            st.session_state.setdefault("_session_log", []).append(rec)
            st.success("Record added to session log.")


# ---------------------------
# Older adult health (frailty & risks)
# ---------------------------

def prisma7_score(answers: list[bool]) -> int:
    return sum(1 for a in answers if a)


def older_adult_ui():
    st.header("Older Adult Health — Frailty & Risk Screener")
    st.caption("Aligned with **UN SDG 3.4**: reduce premature mortality and promote well‑being across ages.")
    disclaimer_box()

    st.markdown("**PRISMA‑7** (Yes = 1):")
    q = [
        "Are you older than 85 years?",
        "Male?",
        "In general, do you have health problems that limit activities?",
        "Do you need someone to help you on a regular basis?",
        "In general, do you have health problems that require you to stay at home?",
        "In case of need, can you count on someone close to you? (No = Yes for scoring)",
        "Do you regularly use a cane, walker, or wheelchair?",
    ]
    ans = []
    cols = st.columns(2)
    for i, question in enumerate(q):
        with cols[i % 2]:
            if i == 5:
                # Reverse scored: No -> 1 point, Yes -> 0
                val = st.selectbox(question, ["Yes", "No"], key=f"p7_{i}")
                ans.append(val == "No")
            else:
                val = st.selectbox(question, ["No", "Yes"], key=f"p7_{i}")
                ans.append(val == "Yes")

    score = prisma7_score(ans)

    st.subheader("Summary")
    oc1, oc2 = st.columns([1, 3])
    with oc1:
        st.metric("PRISMA‑7 score", score)
    with oc2:
        if score >= 3:
            ribbon("Screen suggests possible frailty — consider comprehensive geriatric assessment.", emoji="🧓")
        else:
            ribbon("No frailty suggested by PRISMA‑7 — continue health promotion and routine screening.", emoji="💪")

    st.markdown("**Care prompts** (adapt to local pathways):")
    st.markdown(
        """
        - Review **polypharmacy**, falls risk, vision/hearing, nutrition, mood, and cognition.
        - Encourage **physical activity**, social engagement, and vaccinations per guidelines.
        - Assess **caregiver support**, **advance care planning**, and home safety.
        """
    )

    with st.expander("Save screening record"):
        person_id = st.text_input("Client ID (optional)", key="older_id")
        notes = st.text_area("Notes", key="older_notes")
        if st.button("Add to session log", key="older_add"):
            rec = {
                "timestamp": datetime.utcnow().isoformat(),
                "module": "older_adult",
                "prisma7": score,
                "id": person_id,
                "notes": notes,
            }
            st.session_state.setdefault("_session_log", []).append(rec)
            st.success("Record added to session log.")


# ---------------------------
# Data dashboard (upload & visualize)
# ---------------------------

def example_dataframe(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    data = {
        "module": rng.choice(["child_malnutrition", "pregnancy_loss", "older_adult"], size=n, p=[0.4, 0.3, 0.3]),
        "age": rng.integers(0, 90, size=n),
        "sex": rng.choice(["F", "M"], size=n),
        "muac_cm": np.round(rng.normal(13, 1.5, size=n), 1),
        "prisma7": rng.integers(0, 7, size=n),
        "loss_score": rng.integers(0, 9, size=n),
        "timestamp": pd.date_range(end=pd.Timestamp.utcnow(), periods=n).astype(str),
    }
    return pd.DataFrame(data)


def dashboard_ui():
    st.header("Program Data Dashboard")
    st.caption("Upload CSV data exported from this app or your HMIS. Map columns below.")

    with st.expander("Need sample data?"):
        if st.button("Generate sample dataset"):
            df = example_dataframe(120)
            st.session_state["_dash_df"] = df
            st.success("Sample data generated below.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"]) 
    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state["_dash_df"] = df

    df = st.session_state.get("_dash_df")
    if df is None:
        st.info("Upload a CSV or click 'Generate sample dataset'.")
        return

    st.dataframe(df.head(50))

    # Column mapping
    st.subheader("Column Mapping")
    cols = df.columns.tolist()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        col_module = st.selectbox("Module column", cols, index=cols.index("module") if "module" in cols else 0)
    with c2:
        col_age = st.selectbox("Age column", cols, index=cols.index("age") if "age" in cols else 0)
    with c3:
        col_muac = st.selectbox("MUAC column (optional)", ["<none>"] + cols, index=(cols.index("muac_cm") + 1) if "muac_cm" in cols else 0)
    with c4:
        col_score = st.selectbox("Generic score column (optional)", ["<none>"] + cols, index=(cols.index("loss_score") + 1) if "loss_score" in cols else 0)

    # Summaries
    st.subheader("Quick Stats")
    left, right = st.columns(2)
    with left:
        by_module = df.groupby(col_module).size().reset_index(name="count")
        chart1 = (
            alt.Chart(by_module)
            .mark_bar()
            .encode(x=alt.X(col_module, title="Module"), y=alt.Y("count", title="Records"), tooltip=[col_module, "count"])
        )
        st.altair_chart(chart1, use_container_width=True)

    with right:
        df["_age_bin"] = pd.cut(df[col_age], bins=[-1, 0, 5, 18, 60, 200], labels=["<1y", "1–5y", "6–18y", "19–60y", ">60y"])
        by_age = df.groupby([col_module, "_age_bin"]).size().reset_index(name="count")
        chart2 = (
            alt.Chart(by_age)
            .mark_bar()
            .encode(x=alt.X("_age_bin", title="Age band"), y=alt.Y("count", title="Records"), color=col_module, column=col_module, tooltip=["_age_bin", "count"])
        )
        st.altair_chart(chart2, use_container_width=True)

    # MUAC distribution if available
    if col_muac != "<none>":
        st.subheader("MUAC Distribution")
        muac_df = df[[col_muac]].dropna().rename(columns={col_muac: "muac_cm"})
        chart3 = (
            alt.Chart(muac_df)
            .transform_filter(alt.datum.muac_cm > 0)
            .mark_bar()
            .encode(x=alt.X("muac_cm:Q", bin=alt.Bin(maxbins=20), title="MUAC (cm)"), y=alt.Y("count()", title="Count"))
        )
        st.altair_chart(chart3, use_container_width=True)

    # Generic score distribution
    if col_score != "<none>":
        st.subheader("Score Distribution")
        sd = df[[col_score]].dropna().rename(columns={col_score: "score"})
        chart4 = (
            alt.Chart(sd)
            .mark_bar()
            .encode(x=alt.X("score:Q", bin=alt.Bin(maxbins=20)), y=alt.Y("count()"))
        )
        st.altair_chart(chart4, use_container_width=True)


# ---------------------------
# Session log viewer / export
# ---------------------------

def session_log_ui():
    st.header("Session Log")
    logs = st.session_state.get("_session_log", [])
    if not logs:
        st.info("No records in this session yet. Use the modules to add records.")
        return
    df = pd.DataFrame(logs)
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download session log (CSV)", data=csv, file_name=f"health_navigator_log_{datetime.utcnow().date()}.csv", mime="text/csv")


# ---------------------------
# Sidebar & Navigation
# ---------------------------

st.sidebar.title(APP_TITLE)
st.sidebar.caption(APP_SUBTITLE)

nav = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Child Malnutrition",
        "Pregnancy Loss / Miscarriage",
        "Older Adult Health",
        "Data Dashboard",
        "Session Log",
        "About & Disclaimer",
    ],
)

st.sidebar.markdown("---")
st.sidebar.write("**Version:**", VERSION)

# ---------------------------
# Main Views
# ---------------------------

if nav == "Overview":
    st.title(APP_TITLE)
    st.subheader(APP_SUBTITLE)
    disclaimer_box()

    st.markdown(
        """
        ### How to use
        1. Use the sidebar to open a **screening module**.
        2. Enter available information to compute **basic risk indicators**.
        3. Save records to the **Session Log** and export CSV for program reporting.
        4. Upload data into the **Dashboard** for quick charts.
        
        ### UN SDG Alignment
        - **SDG 3.1**: Maternal health — *Pregnancy Loss* education and early care prompts.
        - **SDG 3.2**: Child health — *Malnutrition screening (MUAC 6–59 months).* 
        - **SDG 3.4**: Healthy aging — *Frailty screening (PRISMA‑7).* 
        
        > Always adapt to **local clinical protocols** and referral pathways.
        """
    )

elif nav == "Child Malnutrition":
    child_malnutrition_ui()

elif nav == "Pregnancy Loss / Miscarriage":
    miscarriage_ui()

elif nav == "Older Adult Health":
    older_adult_ui()

elif nav == "Data Dashboard":
    dashboard_ui()

elif nav == "Session Log":
    session_log_ui()

elif nav == "About & Disclaimer":
    st.header("About")
    st.markdown(
        """
        **UN SDG 3 Health Navigator** is a lightweight Streamlit app for program teams working on **child malnutrition**, **maternal health**, and **older adult well‑being**. 
        It offers **basic screening**, **education**, and **data visualization** to support decision‑making and referrals.
        
        **Important:** The algorithms here are **simplified** and intended for **education and initial screening**. They may not reflect the latest clinical standards in your setting. Always follow **national guidelines**.
        """
    )
    st.markdown("**License:** MIT — use at your own risk. © 2025")
    disclaimer_box()

# Footer
st.markdown("\n\n—\nBuilt with Streamlit to support **UN Sustainable Development Goal 3**.")
