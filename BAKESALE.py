import streamlit as st
import json
import os

# --- Streamlit Page Config ---
st.set_page_config(page_title="Bake Sale Tracker", page_icon="🍰", layout="wide")

DATA_FILE = "bakesale_data.json"

# --- Helper Functions for Persistence ---
def load_data():
    """Load saved data from JSON file if it exists."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
            if "note_counts" in data:
                data["note_counts"] = {int(k): v for k, v in data["note_counts"].items()}
            return data
    else:
        return default_data()

def save_data():
    """Save current session data to JSON file."""
    data = {
        "paan_qty": st.session_state.paan_qty,
        "lemon_qty": st.session_state.lemon_qty,
        "paan_money": st.session_state.paan_money,
        "lemon_money": st.session_state.lemon_money,
        "paan_profit": st.session_state.paan_profit,
        "lemon_profit": st.session_state.lemon_profit,
        "note_counts": st.session_state.note_counts
    }
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

def default_data():
    """Default dataset for app initialization or reset."""
    return {
        "paan_qty": 50,
        "lemon_qty": 30,
        "paan_money": 0,
        "lemon_money": 0,
        "paan_profit": 0,
        "lemon_profit": 0,
        "note_counts": {10: 0, 20: 0, 50: 0, 100: 0, 500: 0, 1000: 0, 5000: 0}
    }

def reset_data():
    """Reset everything to default values."""
    for key, value in default_data().items():
        st.session_state[key] = value
    st.success("🔄 All data has been reset to default values!")

# --- Load Data on Startup ---
saved_data = load_data()

# --- Initialize Session State ---
for key, value in saved_data.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Prices ---
PAAN_COST = 50
PAAN_SELL = 150
LEMON_COST = 50
LEMON_SELL = 150

# --- Sales Functions ---
def sell_paan():
    if st.session_state.paan_qty > 0:
        st.session_state.paan_qty -= 1
        st.session_state.paan_money += PAAN_SELL
        st.session_state.paan_profit += (PAAN_SELL - PAAN_COST)

def sell_lemon():
    if st.session_state.lemon_qty > 0:
        st.session_state.lemon_qty -= 1
        st.session_state.lemon_money += LEMON_SELL
        st.session_state.lemon_profit += (LEMON_SELL - LEMON_COST)

# --- Header ---
st.title("🍰 Bake Sale Dashboard")
st.markdown("Track your bake sale performance, profits, and cash safely — with manual save and reset options.")

# --- Items Section ---
st.header("🧾 Items on Sale")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🥬 Paan")
    st.info(f"**Cost Price:** Rs {PAAN_COST} | **Sale Price:** Rs {PAAN_SELL}")
    st.metric("Quantity Remaining", st.session_state.paan_qty)
    if st.button("Sell Paan", use_container_width=True):
        sell_paan()
        st.rerun()

with col2:
    st.markdown("### 🍋 Lemon Soda")
    st.info(f"**Cost Price:** Rs {LEMON_COST} | **Sale Price:** Rs {LEMON_SELL}")
    st.metric("Quantity Remaining", st.session_state.lemon_qty)
    if st.button("Sell Lemon Soda", use_container_width=True):
        sell_lemon()
        st.rerun()

# --- Financial Calculations ---
paan_invested = 50 * PAAN_COST
lemon_invested = 30 * LEMON_COST
paan_received = st.session_state.paan_money
lemon_received = st.session_state.lemon_money
paan_profit = st.session_state.paan_profit
lemon_profit = st.session_state.lemon_profit
total_profit = paan_profit + lemon_profit

# --- Financial Summary ---
st.header("💰 Financial Summary")
c1, c2, c3 = st.columns(3)
c1.metric("Total Money Invested", f"Rs {paan_invested + lemon_invested}")
c2.metric("Total Money Received", f"Rs {paan_received + lemon_received}")
c3.metric("💸 Total Profit", f"Rs {total_profit}")

# --- Item Breakdown ---
st.markdown("#### 📊 Item-Wise Breakdown")
summary_data = {
    "Item": ["Paan", "Lemon Soda"],
    "Price Invested (Rs)": [paan_invested, lemon_invested],
    "Money Received (Rs)": [paan_received, lemon_received],
    "Profit (Rs)": [paan_profit, lemon_profit],
}
st.dataframe(summary_data, use_container_width=True)

# --- Notes Counter ---
st.header("💵 Cash Notes Counter")

notes = {
    10: "₨10 Notes",
    20: "₨20 Notes",
    50: "₨50 Notes",
    100: "₨100 Notes",
    500: "₨500 Notes",
    1000: "₨1000 Notes",
    5000: "₨5000 Notes"
}

st.markdown("Manage your collected cash denominations:")

for note, label in notes.items():
    cols = st.columns([2, 1, 1])
    with cols[0]:
        st.markdown(f"**{label}:** {st.session_state.note_counts[note]}")
    with cols[1]:
        if st.button(f"+{note}", key=f"add_{note}"):
            st.session_state.note_counts[note] += 1
            st.rerun()
    with cols[2]:
        if st.button(f"-{note}", key=f"sub_{note}"):
            if st.session_state.note_counts[note] > 0:
                st.session_state.note_counts[note] -= 1
            st.rerun()

# --- Total Cash Calculation ---
total_money = sum(note * count for note, count in st.session_state.note_counts.items())
st.success(f"💵 **Total Cash from Notes:** Rs {total_money}")

# --- Save & Reset Buttons ---
st.markdown("---")
save_col, reset_col = st.columns(2)

with save_col:
    if st.button("💾 Save Data to File", use_container_width=True):
        save_data()
        st.success("✅ Data saved successfully to bakesale_data.json!")

with reset_col:
    if st.button("♻️ Reset All Data", use_container_width=True):
        if st.confirm("Are you sure you want to reset all data? This cannot be undone."):
            reset_data()
            st.rerun()
