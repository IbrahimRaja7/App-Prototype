# streamlit_app.py
"""
UN SDG 3 Health Navigator - single-file Streamlit app
- Robust to missing optional packages (graceful degradation)
- Multilingual AI assistant (uses OpenAI if available)
- TTS voice output (gTTS or pyttsx3 if available)
- Onboarding + professional UI, emergency connect, wearable simulation,
  pharmacies, document analysis, quizzes, animations (Lottie if available)
- Designed to avoid ModuleNotFoundError: optional modules are imported with try/except
"""

import os
import re
import time
import random
import importlib
from io import BytesIO
from dataclasses import dataclass
from typing import Dict, List

import streamlit as st

# -------------------------
# SAFE OPTIONAL IMPORTS
# -------------------------
# OpenAI (new or legacy)
_OPENAI_MODE = "none"
_OPENAI_CLIENT = None
_openai_module = None
try:
    # try dynamic import of new OpenAI SDK
    openai_new = importlib.import_module("openai")  # supports both legacy and new in different installs
    # If package has OpenAI class, we'll try to adapt later
    _openai_module = openai_new
    # We will configure usage in ask_ai() based on presence of attributes
    _OPENAI_MODE = "available"
except Exception:
    _OPENAI_MODE = "none"

# gTTS for online TTS
_TTS_GTTS = False
try:
    gtts_mod = importlib.import_module("gtts")
    gTTS = gtts_mod.gTTS
    _TTS_GTTS = True
except Exception:
    _TTS_GTTS = False

# pyttsx3 for offline TTS (fallback)
_TTS_PYTTSX3 = False
try:
    pyttsx3 = importlib.import_module("pyttsx3")
    _TTS_PYTTSX3 = True
except Exception:
    _TTS_PYTTSX3 = False

# deep_translator for translations
_TRANS_OK = False
try:
    dt = importlib.import_module("deep_translator")
    GoogleTranslator = dt.GoogleTranslator
    _TRANS_OK = True
except Exception:
    _TRANS_OK = False

# PyPDF2 for PDF parsing
_PDF_OK = False
PdfReader = None
try:
    pypdf2 = importlib.import_module("PyPDF2")
    PdfReader = pypdf2.PdfReader
    _PDF_OK = True
except Exception:
    _PDF_OK = False

# Lottie (for animations)
_LOTTIE_OK = False
try:
    lottie_mod = importlib.import_module("streamlit_lottie")
    st_lottie = lottie_mod.st_lottie
    _LOTTIE_OK = True
except Exception:
    _LOTTIE_OK = False

# -------------------------
# APP CONFIG & STYLING
# -------------------------
st.set_page_config(page_title="UN SDG 3 Health Navigator", page_icon="🩺", layout="wide")

# Custom CSS (dark inputs + polished cards)
st.markdown(
    """
    <style>
    :root { --brand:#0ea5e9; --accent:#10b981; --bg:#0f1724; --card:#0b1220; --muted:#94a3b8; }
    .stApp { background-color: #061019; }
    .block-container { padding-top: 1rem; padding-left: 1rem; padding-right: 1rem; }
    .card { border-radius: 10px; padding: 1rem; background: linear-gradient(180deg,#071323,#071b2a); border:1px solid rgba(255,255,255,0.03); box-shadow: 0 6px 18px rgba(2,6,23,0.6); color:#e6eef6 }
    .topbar { display:flex; align-items:center; gap:0.75rem; padding:0.5rem; margin-bottom:0.5rem; }
    .brand { font-weight:700; font-size:1.05rem; color: #e6eef6; }
    .chip { display:inline-flex; align-items:center; gap:.4rem; background:#0b2330; border-radius:999px; padding:.25rem .6rem; color:#cde7f5; margin-right:.35rem; }
    /* style inputs */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select, .stTextArea>div>div>textarea {
        background-color: #071323 !important;
        color: #e6eef6 !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 8px !important;
        padding: 8px !important;
    }
    .stButton>button { background: linear-gradient(90deg,#0ea5e9,#10b981); color: #012323; border-radius:8px; padding:6px 10px; }
    .stRadio>div { color: #e6eef6; }
    .alert { padding:.75rem 1rem; border-radius:.65rem; border:1px solid rgba(255,255,255,0.04); background:#04202a; color:#cfeff8; margin-bottom:.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# header/topbar
st.markdown(
    "<div class='topbar'><div class='brand'>UN SDG 3 Health Navigator</div>"
    "<div style='flex:1'></div>"
    "<div class='chip'>By Muhammad Ibrahim Raja</div></div>",
    unsafe_allow_html=True,
)

# -------------------------
# CONSTANTS & DATA
# -------------------------
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
        {"name": "Sehat Pharmacy", "city": "Lahore", "img": "https://via.placeholder.com/120"},
        {"name": "D-Watson Pharmacy", "city": "Islamabad", "img": "https://via.placeholder.com/120"}
    ]),
    "India": CountryMeta(prefix="+91", emergency="112", pharmacies=[
        {"name": "Apollo Pharmacy", "city": "Mumbai", "img": "https://via.placeholder.com/120"},
        {"name": "MedPlus", "city": "Bengaluru", "img": "https://via.placeholder.com/120"},
    ]),
    "USA": CountryMeta(prefix="+1", emergency="911", pharmacies=[
        {"name": "CVS Pharmacy", "city": "New York", "img": "https://via.placeholder.com/120"},
        {"name": "Walgreens", "city": "Chicago", "img": "https://via.placeholder.com/120"},
    ]),
    "UK": CountryMeta(prefix="+44", emergency="999", pharmacies=[
        {"name": "Boots", "city": "London", "img": "https://via.placeholder.com/120"},
    ]),
    "Germany": CountryMeta(prefix="+49", emergency="112", pharmacies=[
        {"name": "Apotheke", "city": "Berlin", "img": "https://via.placeholder.com/120"},
    ]),
}

GLOSSARY = {
    "MUAC": "Mid-Upper Arm Circumference",
    "BMI": "Body Mass Index",
    "SpO₂": "Peripheral Capillary Oxygen Saturation",
    "PRISMA-7": "7-item frailty screening tool",
}

MED_LIBRARY = [
    {"name": "Oral Rehydration Salts (ORS)", "use": "Rehydration for diarrhea", "img": "https://via.placeholder.com/140"},
    {"name": "Paracetamol (Acetaminophen)", "use": "Fever/pain relief", "img": "https://via.placeholder.com/140"},
    {"name": "Iron–Folic Acid (IFA)", "use": "Prevent/treat anemia in pregnancy", "img": "https://via.placeholder.com/140"},
]

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def info_alert(text: str):
    st.markdown(f"<div class='alert'>{text}</div>", unsafe_allow_html=True)

def translate_if_needed(text: str, target_lang: str) -> str:
    """Translate an English text into target language using deep_translator if available."""
    if target_lang == "English" or not _TRANS_OK:
        return text
    try:
        return GoogleTranslator(source="en", target=LANGUAGES[target_lang]).translate(text)
    except Exception:
        return text

def ask_ai(prompt: str, target_lang: str = "English") -> str:
    """
    Ask the AI. Uses whichever OpenAI lib is available.
    If OpenAI not configured, returns a friendly message.
    """
    if _OPENAI_MODE == "none" or _openai_module is None:
        return translate_if_needed(
            "AI not configured. To enable, set OPENAI_API_KEY in Streamlit secrets and install the openai package.",
            target_lang
        )

    # Prepare a short system prompt that emphasizes safety and non-diagnostic output
    system_prompt = (
        "You are a medical information assistant. Provide concise, evidence-informed, plain-language guidance. "
        "Do not provide any personalized diagnosis or prescribe. Always recommend consulting a clinician for definitive diagnosis."
    )

    try:
        # Try to use new-style Chat Completions if available (openai.ChatCompletion)
        if hasattr(_openai_module, "ChatCompletion"):
            resp = _openai_module.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=600,
            )
            answer = resp.choices[0].message["content"] if isinstance(resp.choices[0].message, dict) else resp.choices[0].message
        else:
            # Try attribute 'OpenAI' client style (new SDK)
            if hasattr(_openai_module, "OpenAI"):
                client = _openai_module.OpenAI(api_key=os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", "")))
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"system","content":system_prompt}, {"role":"user","content":prompt}],
                    temperature=0.2,
                )
                # new SDK response shape
                answer = resp.choices[0].message.content
            else:
                return translate_if_needed("OpenAI SDK present but unsupported shape. Please update the package.", target_lang)

        return translate_if_needed(answer, target_lang)
    except Exception as e:
        return translate_if_needed(f"[AI error: {e}]", target_lang)

def tts_stream(text: str, lang: str = "English"):
    """Return bytes-like object for TTS audio or None if TTS unavailable."""
    # Prefer gTTS if available (works online)
    if _TTS_GTTS:
        try:
            code = LANGUAGES.get(lang, "en")
            fp = BytesIO()
            gTTS(text=text, lang=code).write_to_fp(fp)
            fp.seek(0)
            return fp
        except Exception:
            pass
    # Fallback pyttsx3 to produce wav in-memory (if available)
    if _TTS_PYTTSX3:
        try:
            engine = pyttsx3.init()
            # try to save to file-like object - pyttsx3 doesn't support BytesIO directly in many setups,
            # so we will save to a temporary file if necessary.
            tmp_path = "temp_tts_output.mp3"
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            with open(tmp_path, "rb") as f:
                data = BytesIO(f.read())
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            data.seek(0)
            return data
        except Exception:
            pass
    return None

def validate_phone(prefix: str, phone: str) -> bool:
    phone = (phone or "").strip()
    if not phone:
        return False
    # Basic validation: starts with prefix and contains 6-15 digits after plus
    if not phone.startswith(prefix):
        return False
    return re.match(r"^\+\d{6,15}$", phone) is not None

def glossary_chip(keys: List[str]):
    chips = []
    for k in keys:
        v = GLOSSARY.get(k, "")
        chips.append(f"<span style='display:inline-block;background:#062f3a;color:#cfeff8;padding:6px 10px;border-radius:999px;margin-right:6px;font-size:0.9rem'>{k}: {v}</span>")
    st.markdown("".join(chips), unsafe_allow_html=True)

# -------------------------
# STATE & ONBOARDING
# -------------------------
if "user" not in st.session_state:
    st.session_state["user"] = {}

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Onboarding loop (keeps asking until name provided)
if not st.session_state["user"]:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Welcome — Quick setup")
        name = st.text_input("Full name", value="", placeholder="e.g., Amina Khan")
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=25)
        country = st.selectbox("Country", options=list(COUNTRY_INFO.keys()))
        bracket = st.selectbox("Client Health Bracket", options=["Child Malnutrition", "Pregnancy/Miscarriage Risk", "Older Adult Health"])
        language = st.selectbox("Preferred language", options=list(LANGUAGES.keys()), index=0)
        col1, col2 = st.columns([1, 1])
        with col1:
            start_btn = st.button("Start App")
        with col2:
            skip_btn = st.button("Skip (demo user)")
        if start_btn or skip_btn:
            if skip_btn:
                # populate demo user if skipping
                st.session_state["user"] = {"name": "Demo User", "age": 30, "country": country, "bracket": bracket, "language": language}
                st.experimental_rerun()
            else:
                if not name.strip():
                    info_alert("Please enter your name to continue.")
                else:
                    st.session_state["user"] = {"name": name.strip(), "age": age, "country": country, "bracket": bracket, "language": language}
                    st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

user = st.session_state["user"]

# -------------------------
# LAYOUT / NAVIGATION
# -------------------------
st.markdown(f"<div style='margin-bottom:8px'><span style='font-weight:600;color:#cfeff8'>Hello, {user['name']}</span> — <span style='color:#94a3b8'>Country: {user['country']} • Bracket: {user['bracket']}</span></div>", unsafe_allow_html=True)
glossary_chip(["MUAC", "BMI", "SpO₂"])

menu = st.sidebar.radio("Menu", [
    "Overview",
    "Interactive Screening",
    "AI Assistant",
    "Awareness Media",
    "Prevention Animations",
    "Emergency Connect",
    "Wearable Simulation",
    "Pharmacies & Medicines",
    "Document Analysis",
    "Emergency Quiz",
    "About / Disclaimers",
])

# -------------------------
# MENU: Implementation
# -------------------------
if menu == "Overview":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Overview & How to use")
    st.write("""
    This app supports **UN SDG 3**: Good Health and Well-Being.  
    Use the left menu to navigate: screening tools, AI Q&A, awareness media, emergency connect, wearable simulation, pharmacies, and document analysis.
    """)
    info_alert("Educational screening only — not clinical advice. Always consult a healthcare professional for diagnosis and treatment.")
    st.markdown("</div>", unsafe_allow_html=True)

elif menu == "Interactive Screening":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Interactive Screening")
    b = user.get("bracket", "Child Malnutrition")
    if b == "Child Malnutrition":
        st.write("Screening: MUAC (6–59 months) + edema check")
        muac = st.number_input("MUAC (cm)", min_value=0.0, step=0.1)
        edema = st.selectbox("Bilateral pitting edema?", options=["No", "Yes"])
        if st.button("Evaluate MUAC"):
            if muac == 0:
                info_alert("Enter MUAC value to evaluate.")
            else:
                if edema == "Yes" or muac < 11.5:
                    st.error("Severe Acute Malnutrition (SAM) flagged. Urgent referral recommended.")
                elif 11.5 <= muac < 12.5:
                    st.warning("Moderate Acute Malnutrition (MAM). Follow-up & supplementary feeding recommended.")
                else:
                    st.success("No acute malnutrition by MUAC. Continue routine monitoring.")
    elif b == "Pregnancy/Miscarriage Risk":
        st.write("Pregnancy risk indicators (educational)")
        age = st.number_input("Maternal age (years)", min_value=10, max_value=60, value=28)
        prior_losses = st.number_input("Prior pregnancy losses", min_value=0, max_value=10, value=0)
        smoker = st.checkbox("Current smoker / vaping")
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0)
        if st.button("Assess pregnancy risk"):
            score = 0
            if age >= 35: score += 1
            if prior_losses >= 1: score += prior_losses if prior_losses < 4 else 4
            if smoker: score += 1
            if bmi < 18.5 or bmi > 30: score += 1
            if score >= 4:
                st.error("Higher-than-average risk indicators. Recommend early antenatal care and specialist review.")
            elif score >= 2:
                st.warning("Some elevated risk indicators. Address modifiable risks and seek early assessment.")
            else:
                st.success("No major risk indicators flagged here. Maintain healthy preconception habits.")
    else:
        st.write("PRISMA-7 Frailty Screener (Yes=1; Q6 reversed)")
        qs = [
            "Are you older than 85 years?",
            "Male?",
            "Do health problems limit your activities?",
            "Do you need help from someone on a regular basis?",
            "Do health problems require you to stay at home?",
            "Can you count on someone close to you? (Answer No if not) — No = 1 point",
            "Do you regularly use a cane, walker, or wheelchair?",
        ]
        answers = []
        for i, q in enumerate(qs):
            if i == 5:
                val = st.selectbox(q, ["Yes", "No"], key=f"p7_{i}")
                answers.append(val == "No")
            else:
                val = st.selectbox(q, ["No", "Yes"], key=f"p7_{i}")
                answers.append(val == "Yes")
        if st.button("Compute PRISMA-7"):
            total = sum(answers)
            if total >= 3:
                st.warning("Score suggests possible frailty — consider comprehensive geriatric assessment.")
            else:
                st.success("No frailty suggested by PRISMA-7.")
    st.markdown("</div>", unsafe_allow_html=True)

elif menu == "AI Assistant":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("AI Health Assistant (multilingual)")
    st.caption("Educational information only — not a diagnosis.")
    # Show history
    for role, msg in st.session_state["chat_history"][-12:]:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**AI:** {msg}")

    # Chat input (clears automatically)
    user_prompt = st.chat_input("Ask a medical question (short & focused)...")
    if user_prompt:
        st.session_state["chat_history"].append(("user", user_prompt))
        # ask AI
        reply = ask_ai(user_prompt, user.get("language", "English"))
        st.session_state["chat_history"].append(("assistant", reply))
        st.experimental_rerun()

    # Optionally voice
    if st.button("Play last AI reply (voice)"):
        # get last assistant reply
        last = None
        for r, m in reversed(st.session_state["chat_history"]):
            if r == "assistant":
                last = m
                break
        if not last:
            info_alert("No AI reply yet. Ask a question first.")
        else:
            audio = tts_stream(last, user.get("language", "English"))
            if audio:
                st.audio(audio)
            else:
                info_alert("TTS not available. Install gTTS or pyttsx3 to enable voice.")

    st.markdown("</div>", unsafe_allow_html=True)

elif menu == "Awareness Media":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Awareness — images & video")
    st.image("https://via.placeholder.com/800x260.png?text=Child+Malnutrition+Awareness", caption="Child malnutrition is a global challenge.")
    st.video("https://www.youtube.com/watch?v=kl1ujzRidmU")
    st.markdown("</div>", unsafe_allow_html=True)

elif menu == "Prevention Animations":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Prevention Animations")
    if _LOTTIE_OK:
        # Example Lottie URL if available
        try:
            lottie_url = "https://assets5.lottiefiles.com/packages/lf20_k9xY5K.json"
            st_lottie(lottie_url, height=300)
        except Exception:
            st.image("https://via.placeholder.com/800x300.png?text=Handwashing+Animation")
    else:
        st.image("https://via.placeholder.com/800x300.png?text=Handwashing+Animation")
    st.markdown("</div>", unsafe_allow_html=True)

elif menu == "Emergency Connect":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Emergency Connect")
    meta = COUNTRY_INFO.get(user["country"])
    st.markdown(f"**Local emergency number:** {meta.emergency} — **Required prefix for saved contact:** {meta.prefix}")
    trusted = st.text_input("Trusted contact (include +countrycode, e.g., +92XXXXXXXXXX)")
    if st.button("Validate & Save Contact"):
        if validate_phone(meta.prefix, trusted):
            st.success("Trusted contact validated and saved in session.")
            st.session_state["trusted_contact"] = trusted
        else:
            st.error(f"Invalid number. It must start with {meta.prefix} and include digits only (e.g., {meta.prefix}3XXXXXXXXX).")
    if st.button("Simulate Emergency Call (to saved contact)"):
        saved = st.session_state.get("trusted_contact")
        if not saved:
            st.warning("No trusted contact saved — validate one first.")
        else:
            st.info(f"Simulating call to {saved} and local emergency {meta.emergency}... (simulation)")
            # Simulate animation / GIF
            st.image("https://via.placeholder.com/800x200.png?text=Calling...")

    st.markdown("</div>", unsafe_allow_html=True)

elif menu == "Wearable Simulation":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Wearable Band Simulation")
    st.write("Simulated readings from a wearable band. Thresholds trigger alarm simulation.")
    if st.button("Start 10-sec stream"):
        alarm = False
        ph = st.empty()
        for i in range(10):
            hr = random.randint(40, 120)
            temp = round(random.uniform(35.0, 39.5), 1)
            spo2 = random.randint(88, 100)
            with ph.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("Heart Rate (bpm)", hr)
                c2.metric("Temperature (°C)", temp)
                c3.metric("SpO₂ (%)", spo2)
                if hr < 50 or spo2 < 92 or temp > 38:
                    alarm = True
                    st.error("⚠️ Threshold crossed — alarm triggered!")
                    st.image("https://via.placeholder.com/600x120.png?text=ALARM")
            time.sleep(1)
        if alarm:
            st.warning("Auto-call simulation: contacting trusted contact and emergency number...")
    st.markdown("</div>", unsafe_allow_html=True)

elif menu == "Pharmacies & Medicines":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Pharmacies & Medicine Library")
    meta = COUNTRY_INFO.get(user["country"])
    st.write("Nearby sample pharmacies:")
    cols = st.columns(2)
    for i, ph in enumerate(meta.pharmacies):
        with cols[i % 2]:
            st.image(ph["img"], width=110)
            st.markdown(f"**{ph['name']}**")
            st.caption(ph["city"])
    st.markdown("---")
    st.write("Common medicines (educational):")
    cols2 = st.columns(3)
    for i, m in enumerate(MED_LIBRARY):
        with cols2[i % 3]:
            st.image(m["img"], width=110)
            st.markdown(f"**{m['name']}**")
            st.caption(m["use"])
    st.markdown("</div>", unsafe_allow_html=True)

elif menu == "Document Analysis":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Upload & Analyze Health Documents")
    uploaded = st.file_uploader("Upload CSV / TXT / PDF (small files recommended)")
    if uploaded:
        name = uploaded.name.lower()
        if name.endswith(".csv"):
            try:
                import pandas as pd
                df = pd.read_csv(uploaded)
                st.dataframe(df.head(50))
                st.write(df.describe(include="all"))
            except Exception as e:
                st.error(f"CSV parsing error: {e}")
        elif name.endswith(".txt"):
            text = uploaded.read().decode("utf-8", errors="ignore")
            st.text_area("File content (first 4000 chars)", text[:4000], height=240)
        elif name.endswith(".pdf"):
            if _PDF_OK and PdfReader is not None:
                try:
                    reader = PdfReader(uploaded)
                    pages = [p.extract_text() or "" for p in reader.pages]
                    text = "\n".join(pages)[:4000]
                    st.text_area("Extracted text (first 4k chars)", text, height=240)
                except Exception as e:
                    st.error(f"PDF parsing error: {e}")
            else:
                st.warning("PDF parsing not available. Install PyPDF2 to enable.")
        else:
            st.info("Unsupported file type. Use CSV / TXT / PDF.")
    st.markdown("</div>", unsafe_allow_html=True)

elif menu == "Emergency Quiz":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Emergency Preparedness Quiz")
    q = st.radio("Someone collapses and is unresponsive. What is the first step?", [
        "Shake them and shout loudly",
        "Check responsiveness & breathing and call emergency if absent",
        "Give them water"
    ])
    if st.button("Submit Answer"):
        if q == "Check responsiveness & breathing and call emergency if absent":
            st.success("Correct — ensure airway & breathing, then call emergency services.")
        else:
            st.error("Not correct. First check responsiveness & breathing; call for help if needed.")
    st.markdown("</div>", unsafe_allow_html=True)

elif menu == "About / Disclaimers":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("About & Disclaimers")
    st.write("This educational app supports UN SDG 3 (Good Health & Well-Being). It was designed by Muhammad Ibrahim Raja.")
    st.warning("Not medical advice. Use for education and screening guidance only. Always consult trained clinicians for diagnosis and treatment.")
    # show available features vs missing modules
    st.markdown("**Runtime feature availability:**")
    st.write(f"- OpenAI available: {_OPENAI_MODE != 'none'}")
    st.write(f"- Translation (deep_translator): {_TRANS_OK}")
    st.write(f"- gTTS TTS: {_TTS_GTTS}")
    st.write(f"- pyttsx3 TTS fallback: {_TTS_PYTTSX3}")
    st.write(f"- PDF parsing (PyPDF2): {_PDF_OK}")
    st.write(f"- Lottie animations (streamlit_lottie): {_LOTTIE_OK}")
    st.markdown("</div>", unsafe_allow_html=True)

# End of app
Notes & next steps

Save the file as streamlit_app.py at repository root.

To enable full AI and TTS features, add these to requirements.txt (recommended):

nginx
Copy
Edit
streamlit
openai
gTTS
deep-translator
PyPDF2
streamlit-lottie
pyttsx3
pandas
 
