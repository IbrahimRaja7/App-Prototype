import streamlit as st
import pandas as pd
import numpy as np
import time
import random

# -------------------------
# App Config
# -------------------------
st.set_page_config(page_title="UN SDG 3 Health App", page_icon="🩺", layout="wide")

# -------------------------
# Utilities
# -------------------------
def disclaimer():
    st.warning("⚠️ This app is for **educational and awareness purposes only**. It does not provide medical advice. Always consult a healthcare professional.")

def ribbon(text, emoji="ℹ️"):
    st.markdown(f"<div style='background:#f0fdf4;padding:0.5rem 1rem;border-radius:0.5rem;font-weight:500;'>{emoji} {text}</div>", unsafe_allow_html=True)

# -------------------------
# Onboarding
# -------------------------
if "user" not in st.session_state:
    st.session_state.user = {}

st.title("🌍 UN SDG 3 Health Navigator")
st.caption("Designed by **Muhammad Ibrahim Raja** | Supporting Sustainable Development Goal 3: Good Health and Well-Being")

if not st.session_state.user:
    st.subheader("👤 Let's get to know you")
    name = st.text_input("Your Name")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    country = st.text_input("Country")
    bracket = st.selectbox("Select your health bracket", ["Child Malnutrition", "Pregnancy/Miscarriage Risk", "Older Adult Health"])
    if st.button("Continue"):
        if name and age and country and bracket:
            st.session_state.user = {"name": name, "age": age, "country": country, "bracket": bracket}
            st.success(f"Welcome {name}, let's continue.")
        else:
            st.error("Please fill all fields to continue.")
    st.stop()

user = st.session_state.user
st.sidebar.success(f"Logged in as {user['name']} ({user['bracket']})")

# -------------------------
# Navigation
# -------------------------
section = st.sidebar.radio("Navigate", [
    "Information Hub",
    "Interactive Screening",
    "AI Assistant",
    "Awareness (Images/Videos)",
    "Prevention Animations",
    "Emergency Connect",
    "Wearable Simulation",
    "Emergency Quiz",
    "Document Analysis",
    "About"
])

# -------------------------
# Sections
# -------------------------
if section == "Information Hub":
    st.header("ℹ️ Information Hub")
    if user["bracket"] == "Child Malnutrition":
        st.write("Children under 5 remain at risk of malnutrition worldwide. Early detection and proper nutrition can save lives.")
    elif user["bracket"] == "Pregnancy/Miscarriage Risk":
        st.write("Pregnancy health is vital to prevent miscarriages. Risk factors include maternal age, infections, chronic diseases, and lifestyle.")
    else:
        st.write("Older adults face risks such as frailty, chronic disease, and falls. Preventive care and early detection improve well-being.")
    disclaimer()

elif section == "Interactive Screening":
    st.header("📝 Screening")
    if user["bracket"] == "Child Malnutrition":
        muac = st.number_input("Enter MUAC (cm)", min_value=0.0, step=0.1)
        if muac:
            if muac < 11.5:
                ribbon("Severe Acute Malnutrition — seek urgent care.", "🚨")
            elif muac < 12.5:
                ribbon("Moderate Malnutrition — requires follow-up.", "⚠️")
            else:
                ribbon("No acute malnutrition detected.", "✅")
    elif user["bracket"] == "Pregnancy/Miscarriage Risk":
        st.write("Answer a few lifestyle questions:")
        smoker = st.checkbox("Do you smoke?")
        bmi = st.number_input("Your BMI", min_value=10.0, max_value=50.0, step=0.1)
        if st.button("Evaluate"):
            if smoker or bmi < 18.5 or bmi > 30:
                ribbon("Some risk factors detected — seek early antenatal care.", "⚠️")
            else:
                ribbon("No major risks flagged.", "✅")
    else:
        st.write("PRISMA-7 Frailty screening:")
        q1 = st.selectbox("Do you often need help in daily activities?", ["No", "Yes"])
        if q1 == "Yes":
            ribbon("Possible frailty detected — consider geriatric assessment.", "👵")

elif section == "AI Assistant":
    st.header("🤖 Health Q&A Assistant")
    query = st.text_input("Ask me a question about your health bracket")
    if query:
        # Placeholder for AI API integration
        st.write(f"AI response (simulated): '{query}' is an important question. Please consult your doctor for personalized advice.")

elif section == "Awareness (Images/Videos)":
    st.header("📸 Awareness Media")
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/0a/Child_malnutrition.jpg", caption="Child malnutrition remains a global issue.")
    st.video("https://www.youtube.com/watch?v=kl1ujzRidmU")

elif section == "Prevention Animations":
    st.header("🎬 Preventive Health Animations")
    st.write("Animation: Proper handwashing saves lives.")
    st.image("https://media.giphy.com/media/3o6ZsZknK9PNg5FfEA/giphy.gif")

elif section == "Emergency Connect":
    st.header("🚨 Emergency Connect")
    st.write(f"In {user['country']}, common emergency number is: 112 (verify locally).")
    phone = st.text_input("Enter trusted contact phone number")
    if st.button("Verify & Simulate Call"):
        st.success(f"Calling {phone} and local emergency number... (simulation)")
        st.image("https://media.giphy.com/media/l0ExvMq5G5IsnXzRS/giphy.gif")

elif section == "Wearable Simulation":
    st.header("⌚ Wearable Band Simulation")
    hr = random.randint(40, 120)
    temp = round(random.uniform(35.0, 39.5), 1)
    st.metric("Heart Rate", f"{hr} bpm")
    st.metric("Temperature", f"{temp} °C")
    if hr < 50 or hr > 110:
        st.error("🚨 Abnormal heart rate detected! Triggering alarm...")
        st.image("https://media.giphy.com/media/3oEjI6SIIHBdRxXI40/giphy.gif")

elif section == "Emergency Quiz":
    st.header("🧠 Emergency Preparedness Quiz")
    q = st.radio("You see someone faint. What’s your first response?", ["Run away", "Check responsiveness and breathing", "Give water immediately"])
    if st.button("Submit Answer"):
        if q == "Check responsiveness and breathing":
            st.success("Correct! Always check responsiveness and breathing first.")
        else:
            st.error("Not correct. Remember: check responsiveness and breathing.")

elif section == "Document Analysis":
    st.header("📂 Health Document Analysis")
    file = st.file_uploader("Upload a health-related CSV/PDF")
    if file and file.name.endswith(".csv"):
        df = pd.read_csv(file)
        st.write("Preview:", df.head())
        st.write("Basic Stats:", df.describe())
    elif file:
        st.info("PDF/Text support coming soon.")

elif section == "About":
    st.header("ℹ️ About")
    st.write("This app supports **UN Sustainable Development Goal 3** (Good Health & Well-Being). It was designed by **Muhammad Ibrahim Raja** to educate, screen, and raise awareness for child malnutrition, maternal health, and older adult well-being.")
    disclaimer()
