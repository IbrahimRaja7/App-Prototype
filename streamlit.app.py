import streamlit as st
import pandas as pd
import numpy as np
import random
from gtts import gTTS
from io import BytesIO
import openai
from deep_translator import GoogleTranslator

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="UN SDG 3 Health Navigator", page_icon="🩺", layout="wide")

# OpenAI API Key (set in .streamlit/secrets.toml)
openai.api_key = st.secrets.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

# Languages Supported
LANGUAGES = {
    "English": "en",
    "Urdu": "ur",
    "German": "de",
    "Arabic": "ar",
    "Spanish": "es"
}

# Country info with emergency numbers & pharmacies
COUNTRY_INFO = {
    "Pakistan": {"prefix": "+92", "emergency": "15", "pharmacies": ["D-Watson Pharmacy", "Shaheen Chemist"]},
    "India": {"prefix": "+91", "emergency": "112", "pharmacies": ["Apollo Pharmacy", "MedPlus"]},
    "USA": {"prefix": "+1", "emergency": "911", "pharmacies": ["CVS Pharmacy", "Walgreens"]},
    "UK": {"prefix": "+44", "emergency": "999", "pharmacies": ["Boots", "Superdrug"]},
    "Germany": {"prefix": "+49", "emergency": "112", "pharmacies": ["Apotheke", "dm"]},
}

# -------------------------
# UTILITIES
# -------------------------
def disclaimer():
    st.warning("⚠️ Disclaimer: This app is for **educational and awareness purposes only**. It does not replace professional medical advice. Consult a healthcare provider for personal medical concerns.")

def ai_response(prompt, lang="English"):
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional medical assistant providing evidence-based and reliable health information aligned with WHO and UN SDG 3."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = completion.choices[0].message["content"]
        if lang != "English":
            answer = GoogleTranslator(source="en", target=LANGUAGES[lang]).translate(answer)
        return answer
    except Exception as e:
        return f"[Error fetching AI response: {str(e)}]"

def text_to_speech(text, lang="English"):
    lang_code = LANGUAGES.get(lang, "en")
    tts = gTTS(text=text, lang=lang_code)
    fp = BytesIO()
    tts.write_to_fp(fp)
    return fp

# -------------------------
# ONBOARDING
# -------------------------
if "user" not in st.session_state:
    st.session_state.user = {}

st.title("🌍 UN SDG 3 Health Navigator")
st.caption("Designed by **Muhammad Ibrahim Raja** | Supporting **Sustainable Development Goal 3: Good Health and Well-Being**")

if not st.session_state.user:
    st.subheader("👤 Let's get to know you")
    name = st.text_input("Your Name")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    country = st.selectbox("Country", list(COUNTRY_INFO.keys()))
    bracket = st.selectbox("Select your health bracket", ["Child Malnutrition", "Pregnancy/Miscarriage Risk", "Older Adult Health"])
    language = st.selectbox("Preferred Language", list(LANGUAGES.keys()))
    if st.button("Continue"):
        if name and age and country and bracket:
            st.session_state.user = {"name": name, "age": age, "country": country, "bracket": bracket, "language": language}
            st.success(f"✅ Welcome {name}, let's continue.")
        else:
            st.error("⚠️ Please fill all fields to continue.")
    st.stop()

user = st.session_state.user

# -------------------------
# NAVIGATION MENU
# -------------------------
menu = st.sidebar.radio("☰ Navigation", [
    "Information Hub",
    "AI Assistant",
    "Awareness Media",
    "Prevention Animations",
    "Emergency Connect",
    "Wearable Simulation",
    "Nearby Pharmacies",
    "Emergency Quiz",
    "Document Analysis",
    "About"
])

# -------------------------
# SECTIONS
# -------------------------
if menu == "Information Hub":
    st.header("ℹ️ Information Hub")
    if user["bracket"] == "Child Malnutrition":
        st.write("**Child Malnutrition**: Affects nearly 45 million children under 5 worldwide. Early detection & balanced nutrition save lives.")
        st.info("📏 MUAC (*Mid-Upper Arm Circumference*): <11.5 cm indicates **Severe Malnutrition**")
    elif user["bracket"] == "Pregnancy/Miscarriage Risk":
        st.write("**Pregnancy & Miscarriage Risk**: Miscarriage affects ~10–20% of known pregnancies. Risk factors: maternal age, infections, chronic disease, smoking, obesity.")
        st.info("📊 BMI (*Body Mass Index*): Below 18.5 or above 30 may increase risks.")
    else:
        st.write("**Older Adult Health**: 1 in 3 adults >65 fall each year. Frailty & chronic disease need regular screening.")
        st.info("🧮 PRISMA-7 screening helps detect frailty early.")
    disclaimer()

elif menu == "AI Assistant":
    st.header("🤖 AI Health Assistant")
    q = st.text_input("Ask a medical question")
    if st.button("Ask") and q:
        ans = ai_response(q, user['language'])
        st.success(ans)
        audio_fp = text_to_speech(ans, user['language'])
        st.audio(audio_fp)
        st.session_state.last_question = q
        st.experimental_rerun()

elif menu == "Awareness Media":
    st.header("📸 Awareness Media")
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/0a/Child_malnutrition.jpg", caption="Child malnutrition is a persistent issue.")
    st.video("https://www.youtube.com/watch?v=kl1ujzRidmU")

elif menu == "Prevention Animations":
    st.header("🎬 Preventive Health Animations")
    st.write("🧼 Proper handwashing prevents infections.")
    st.image("https://media.giphy.com/media/3o6ZsZknK9PNg5FfEA/giphy.gif")

elif menu == "Emergency Connect":
    st.header("🚨 Emergency Connect")
    country_data = COUNTRY_INFO[user['country']]
    st.write(f"In {user['country']}, the emergency number is: **{country_data['emergency']}**")
    phone = st.text_input("Enter trusted contact number", value=country_data['prefix'])
    if st.button("Simulate Emergency Call"):
        if phone.startswith(country_data['prefix']):
            st.success(f"📞 Calling {phone} and emergency number {country_data['emergency']}... (simulation)")
            st.image("https://media.giphy.com/media/l0ExvMq5G5IsnXzRS/giphy.gif")
        else:
            st.error(f"Phone must start with {country_data['prefix']}")

elif menu == "Wearable Simulation":
    st.header("⌚ Wearable Band Simulation")
    hr = random.randint(40, 120)
    temp = round(random.uniform(35.0, 39.5), 1)
    st.metric("❤️ Heart Rate", f"{hr} bpm")
    st.metric("🌡️ Temperature", f"{temp} °C")
    if hr < 50 or hr > 110:
        st.error("🚨 Abnormal heart rate detected! Alarm triggered.")
        st.image("https://media.giphy.com/media/3oEjI6SIIHBdRxXI40/giphy.gif")

elif menu == "Nearby Pharmacies":
    st.header("🏥 Nearby Pharmacies & Medicines")
    pharmacies = COUNTRY_INFO[user['country']]["pharmacies"]
    st.subheader("📍 Pharmacies in your region:")
    for p in pharmacies:
        st.write(f"- {p}")
    st.subheader("💊 Common Medicines")
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/07/Paracetamol-sample.jpg", caption="Paracetamol — Pain relief & fever")
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/8a/Amoxicillin-500mg-capsules.jpg", caption="Amoxicillin — Antibiotic")

elif menu == "Emergency Quiz":
    st.header("🧠 Emergency Preparedness Quiz")
    q = st.radio("You see someone faint. What’s your first response?", [
        "Run away",
        "Check responsiveness and breathing",
        "Give water immediately"
    ])
    if st.button("Submit Answer"):
        if q == "Check responsiveness and breathing":
            st.success("✅ Correct! Always check responsiveness and breathing first.")
        else:
            st.error("❌ Incorrect. Remember: check responsiveness and breathing first.")

elif menu == "Document Analysis":
    st.header("📂 Health Document Analysis")
    file = st.file_uploader("Upload a health-related CSV")
    if file and file.name.endswith(".csv"):
        df = pd.read_csv(file)
        st.write("📊 File Preview:")
        st.dataframe(df.head())
        st.write("📈 Summary Statistics:")
        st.write(df.describe())

elif menu == "About":
    st.header("ℹ️ About")
    st.write("""
    This app supports **UN Sustainable Development Goal 3** (Good Health & Well-Being).  
    It was designed by **Muhammad Ibrahim Raja** to educate, screen, and raise awareness  
    for **child malnutrition**, **maternal health**, and **older adult well-being**.
    """)
    disclaimer()
