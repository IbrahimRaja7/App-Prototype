import streamlit as st

# --- Streamlit Page Config ---
st.set_page_config(page_title="Bake Sale Tracker", page_icon="🍰", layout="wide")

st.title("🍰 Bake Sale Dashboard")

st.markdown("Keep track of your bake sale items, sales, and cash notes easily!")

# --- Initialize session state ---
if "paan_qty" not in st.session_state:
    st.session_state.paan_qty = 50
if "lemon_qty" not in st.session_state:
    st.session_state.lemon_qty = 30
if "paan_money" not in st.session_state:
    st.session_state.paan_money = 0
if "lemon_money" not in st.session_state:
    st.session_state.lemon_money = 0
if "paan_profit" not in st.session_state:
    st.session_state.paan_profit = 0
if "lemon_profit" not in st.session_state:
    st.session_state.lemon_profit = 0

# --- Prices ---
PAAN_COST = 50
PAAN_SELL = 150
LEMON_COST = 50
LEMON_SELL = 150

# --- Functions for Sales ---
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

# --- Layout for Items ---
st.header("🧾 Items on Sale")

item_col1, item_col2 = st.columns(2)

with item_col1:
    st.markdown("### 🥬 Paan")
    st.info(f"**Cost Price:** Rs {PAAN_COST} | **Sale Price:** Rs {PAAN_SELL}")
    st.metric("Quantity Remaining", st.session_state.paan_qty)
    st.button("Sell Paan", on_click=sell_paan, use_container_width=True)
    st.markdown("---")

with item_col2:
    st.markdown("### 🍋 Lemon Soda")
    st.info(f"**Cost Price:** Rs {LEMON_COST} | **Sale Price:** Rs {LEMON_SELL}")
    st.metric("Quantity Remaining", st.session_state.lemon_qty)
    st.button("Sell Lemon Soda", on_click=sell_lemon, use_container_width=True)
    st.markdown("---")

# --- Financial Calculations ---
paan_invested = 50 * PAAN_COST
lemon_invested = 30 * LEMON_COST
paan_received = st.session_state.paan_money
lemon_received = st.session_state.lemon_money
paan_profit = st.session_state.paan_profit
lemon_profit = st.session_state.lemon_profit
total_profit = paan_profit + lemon_profit

# --- Financial Summary Section ---
st.header("💰 Financial Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)
summary_col1.metric("Total Money Invested", f"Rs {paan_invested + lemon_invested}")
summary_col2.metric("Total Money Received", f"Rs {paan_received + lemon_received}")
summary_col3.metric("💸 Total Profit", f"Rs {total_profit}")

# --- Detailed Table ---
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

if "note_counts" not in st.session_state:
    st.session_state.note_counts = {note: 0 for note in notes.keys()}

total_money = 0

st.markdown("Manage your collected cash denominations:")

for note, label in notes.items():
    colA, colB, colC = st.columns([2, 1, 1])
    with colA:
        st.write(f"**{label}:** {st.session_state.note_counts[note]}")
    with colB:
        if st.button(f"+{note}", key=f"add_{note}"):
            st.session_state.note_counts[note] += 1
    with colC:
        if st.button(f"-{note}", key=f"sub_{note}"):
            if st.session_state.note_counts[note] > 0:
                st.session_state.note_counts[note] -= 1
    total_money += note * st.session_state.note_counts[note]

st.success(f"💵 **Total Cash from Notes:** Rs {total_money}")


