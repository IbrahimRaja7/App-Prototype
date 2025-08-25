# streamlit_app.py
import os
import re
import time
import random
from io import BytesIO
from dataclasses import dataclass
from typing import Dict, List

import streamlit as st
import pandas as pd
import numpy as np

# -------- Optional deps (graceful fallback if missing) --------
try:
    from gtts import gTTS
    _TTS_OK = True
except Exception:
    _TTS_OK = False

try:
    from deep_translator import GoogleTranslator
    _TRANS_OK = True
except Exception:
    _TRANS_OK = False

try:
    from PyPDF2 import PdfReader
    _PDF_OK = True
except Exception:
    _PDF_OK = False

# --- OpenAI (new SDK with legacy fallback; otherwise disabled) ---
_OPENAI_MODE = "none"
_OPENAI_CLIENT = None
try:
    from openai import OpenAI  # new SDK
    _OPENAI_CLIENT = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")))
    _OPENAI_MODE = "new"
except Exception:
    try:
        import importlib
        openai = importlib.import_module("openai")  # legacy SDK
        openai.api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        _OPENAI_MODE = "legacy"
        _OPENAI_CLIENT = None
    except Exception:
        _OPENAI_MODE = "none"
        _OPENAI_CLIENT = None

# =========================
# App Config & Styling
# =========================
st.set_page_config(page_title="UN SDG 3 Health Navigator", page_icon="🩺", layout="wide")

CUSTOM_CSS = """
<style>
:root { --brand:#0ea5e9; --accent:#10b981; --warn:#f59e0b; --danger:#ef4444; }
.block-container { padding-top: 0.8rem; }
.topbar { position: sticky; top:0; z-index:999; background:white; border-bottom:1px solid #eef2f6;
          padding:.6rem .75rem; display:flex; align-items:center; gap:.75rem; }
.brand { font-weight:700; font-size:1.1rem; color:#0f172a; }
.badge { background:#ecfeff; border:1px solid #cffafe; color:#0369a1; border-radius:999px; padding:.2rem .6rem; font-size:.8rem; }
.card { border:1px solid #e5e7eb; border-radius:.9rem; padding:1rem; background:white; box-shadow:0 4px 16px rgba(2,6,23,.03); }
.alert { padding:.75rem 1rem; border-radius:.65rem; border:1px solid #e5e7eb; }
.alert.info{ background:#f0f9ff; border-color:#bae6fd; }
.alert.warn{ background:#fffbeb; border-color:#fde68a; }
.alert.danger{ background:#fef2f2; border-color:#fecaca; }
.chip { display:inline-flex; align-items:center; gap:.3rem; background:#f1f5f9; border:1px solid #e5e7eb;
        border-radius:999px; padding:.15rem .6rem; margin-right:.25rem; font-size:.8rem; }
.small { font-size:.9rem; color:#475569; }
.sidebar-title { font-weight:700; margin-bottom:.25rem; }
hr.div { border:0; height:1px; background:#eff3f8; margin:.75rem 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    "<div class='topbar'><span class='brand'>UN SDG 3 Health Navigator</span>"
    "<span class='badge'>By Muhammad Ibrahim Raja</span></div>",
    unsafe_allow_html=True
)

# =========================
# Data
# =========================
LANGUAGES = {
    "English": "en",
    "Urdu": "ur",
    "German": "de",
    "Arabic": "ar",
    "Spanish": "es",
}

@dataclass
class CountryMeta:
    prefix: str
    emergency: str
    pharmacies: List[Dict[str, str]]

COUNTRY_INFO: Dict[str, CountryMeta] = {
    "Pakistan": CountryMeta(prefix="+92", emergency="15", pharmacies=[
        {"name": "Sehat Pharmacy", "city": "Lahore",
         "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Pharmacy_symbol.svg/320px-Pharmacy_symbol.svg.png"},
        {"name": "D-Watson", "city": "Islamabad",
         "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Pill_icon.svg/320px-Pill_icon.svg.png"},
    ]),
    "India": CountryMeta(prefix="+91", emergency="112", pharmacies=[
        {"name": "Apollo Pharmacy", "city": "Mumbai",
         "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Pharmacy.svg/320px-Pharmacy.svg.png"},
        {"name": "MedPlus", "city": "Bengaluru",
         "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/Medicine_Capsule.svg/320px-Medicine_Capsule.svg.png"},
    ]),
    "USA": CountryMeta(prefix="+1", emergency="911", pharmacies=[
        {"name": "CVS Pharmacy", "city": "New York",
         "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Pictograms-nps-medicine.svg/320px-Pictograms-nps-medicine.svg.png"},
        {"name": "Walgreens", "city": "Chicago",
         "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Pharmaceutical_Drugs.svg/320px-Pharmaceutical_Drugs.svg.png"},
    ]),
    "UK": CountryMeta(prefix="+44", emergency="999", pharmacies=[
        {"name": "Boots", "city": "London",
         "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Pill_bottle_icon.svg/320px-Pill_bottle_icon.svg.png"},
    ]),
    "Germany": CountryMeta(prefix="+49", emergency="112", pharmacies=[
        {"name": "Apotheke", "city": "Berlin",
         "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Pharmacy_symbol.svg/320px-Pharmacy_symbol.svg.png"},
    ]),
}

MED_LIBRARY = [
    {"name": "Oral Rehydration Salts (ORS)", "use": "Treat dehydration (diarrhea)",
     "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Oral_rehydration_salts.jpg/320px-Oral_rehydration_salts.jpg"},
    {"name": "Paracetamol (Acetaminophen)", "use": "Fever / pain relief",
     "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Acetaminophen-3D-balls.png/320px-Acetaminophen-3D-balls.png"},
    {"name": "Iron–Folic Acid (IFA)", "use": "Pregnancy anemia prevention",
     "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Iron_pills.jpg/320px-Iron_pills.jpg"},
]

GLOSSARY = {
    "MUAC": "Mid-Upper Arm Circumference",
    "BMI": "Body Mass Index",
    "SpO₂": "Peripheral Capillary Oxygen Saturation",
    "PRISMA-7": "Program of Research to Integrate Services for the Maintenance of Autonomy – 7-item frailty screen",
}

# =========================
# Helpers
# =========================
def alert(text: str, level: str = "info"):
    st.markdown(f"<div class='alert {level}'>{text}</div>", unsafe_allow_html=True)

def glossary_inline(keys: List[str]):
    chips = []
    for k in keys:
        v = GLOSSARY.get(k, "")
        chips.append(f"<span class='chip' title='{v}'>{k}: {v}</span>")
    st.markdown(" ".join(chips), unsafe_allow_html=True)

def translate_text(text: str, target_lang: str) -> str:
    if target_lang == "English":
        return text
    if _TRANS_OK:
        try:
            return GoogleTranslator(source="en", target=LANGUAGES[target_lang]).translate(text)
        except Exception:
            return text  # fallback silently
    return text

def ask_ai(prompt: str, lang: str) -> str:
    """Call OpenAI (new/legacy). If not configured, return a friendly message."""
    system_msg = (
        "You are a medical information assistant. Provide concise, evidence-informed, plain-language guidance. "
        "Always include a short disclaimer and advise consulting a clinician for diagnosis/treatment."
    )
    try:
        if _OPENAI_MODE == "new" and _OPENAI_CLIENT is not None:
            resp = _OPENAI_CLIENT.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system_msg},
                          {"role": "user", "content": prompt}],
                temperature=0.2,
            )
            answer = resp.choices[0].message.content
        elif _OPENAI_MODE == "legacy":
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": system_msg},
                          {"role": "user", "content": prompt}],
                temperature=0.2,
            )
            answer = completion.choices[0].message["content"]
        else:
            return "AI not configured. Add OPENAI_API_KEY in Streamlit secrets or environment to enable answers."
        return translate_text(answer, lang)
    except Exception as e:
        return f"[AI error: {e}]"

def tts_bytes(text: str, lang: str):
    """Return MP3 bytes for text if gTTS available; else None."""
    if not _TTS_OK:
        return None
    try:
        code = LANGUAGES.get(lang, "en")
        tts = gTTS(text=text, lang=code)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception:
        return None

def validate_phone(prefix: str, phone: str) -> bool:
    phone = phone.strip()
    return phone.startswith(prefix) and re.match(r"^\+\d{6,15}$", phone) is not None

# =========================
# State
# =========================
if "user" not in st.session_state:
    st.session_state.user = {}
if "chat" not in st.session_state:
    st.session_state.chat = []
if "q" not in st.session_state:
    st.session_state.q = ""

# =========================
# Onboarding
# =========================
st.title("🌍 UN SDG 3 Health Navigator")
st.caption("Designed by **Muhammad Ibrahim Raja** • Supports **SDG 3: Good Health & Well-Being**")

if not st.session_state.user:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("👤 Let’s set you up")
        name = st.text_input("Full Name")
        age = st.number_input("Age (years)", 0, 120, 25)
        country = st.selectbox("Country", list(COUNTRY_INFO.keys()))
        bracket = st.selectbox("Client Health Bracket", [
            "Child Malnutrition", "Pregnancy/Miscarriage Risk", "Older Adult Health"
        ])
        language = st.selectbox("Preferred Language", list(LANGUAGES.keys()))
        if st.button("Start"):
            if name and country:
                st.session_state.user = {"name": name, "age": age, "country": country,
                                         "bracket": bracket, "language": language}
                alert(f"Welcome {name}! You selected **{bracket}** in **{country}**.")
            else:
                alert("Please fill all required fields.", "warn")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

user = st.session_state.user

# =========================
# Sidebar Navigation
# =========================
st.sidebar.markdown("<div class='sidebar-title'>☰ Navigation</div>", unsafe_allow_html=True)
menu = st.sidebar.radio(
    label="",
    options=[
        "Home",
        "Information Hub",
        "Interactive Screening",
        "AI Assistant",
        "Awareness Media",
        "Prevention Animations",
        "Emergency Connect",
        "Wearable Band",
        "Pharmacies & Medicines",
        "Emergency Quiz",
        "Document Analysis",
        "About & Disclaimers",
    ],
    index=0,
)

# header chips
st.markdown(
    f"<span class='chip'>User: {user['name']}</span> "
    f"<span class='chip'>Country: {user['country']}</span> "
    f"<span class='chip'>Bracket: {user['bracket']}</span> "
    f"<span class='chip'>Language: {user['language']}</span>",
    unsafe_allow_html=True,
)

# =========================
# Screens
# =========================
if menu == "Home":
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Overview")
        st.write(
            "A professional, multilingual health interface aligned with **UN SDG 3**. "
            "Use the sidebar to navigate modules, from education and screening to AI answers and emergency tools."
        )
        glossary_inline(["MUAC", "BMI", "SpO₂", "PRISMA-7"])
        alert("Educational use only — not medical advice.", "warn")
        st.markdown("</div>", unsafe_allow_html=True)

elif menu == "Information Hub":
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Information Hub")
        b = user["bracket"]
        if b == "Child Malnutrition":
            st.markdown(
                "- **Child Malnutrition (6–59 months):** Early detection via **MUAC** and clinical signs reduces mortality. "
                "Balanced diet (diverse food groups), vaccination, vitamin A, deworming, and WASH are key.\n"
                "- **MUAC thresholds:** <11.5 cm **Severe Acute Malnutrition (SAM)**; 11.5–<12.5 cm **Moderate Acute Malnutrition (MAM)**."
            )
        elif b == "Pregnancy/Miscarriage Risk":
            st.markdown(
                "- **Pregnancy Health:** Risks for loss include advanced age, infections, uncontrolled chronic disease, "
                "smoking, alcohol, and extremes of **BMI**. Early antenatal care, folic acid, and screening mitigate risk.\n"
                "- **Red flags:** Heavy bleeding, severe pain, fever — seek urgent care."
            )
        else:
            st.markdown(
                "- **Older Adult Health:** Frailty, polypharmacy, falls, cognitive decline. Regular review of meds, "
                "strength & balance training, vision checks, and home safety reduce harm.\n"
                "- **PRISMA-7** helps screen for frailty in primary care."
            )
        glossary_inline(["MUAC", "BMI"])
        st.markdown("</div>", unsafe_allow_html=True)

elif menu == "Interactive Screening":
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Interactive Screening")
        b = user["bracket"]
        if b == "Child Malnutrition":

