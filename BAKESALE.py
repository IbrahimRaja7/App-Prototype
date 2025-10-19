import streamlit as st
import json
import os

# --- Page config ---
st.set_page_config(page_title="Bake Sale Tracker", page_icon="🍰", layout="wide")

DATA_FILE = "bakesale_data.json"

# --- Helper functions ---
def default_data():
    return {
        "paan_qty": 50,
        "lemon_qty": 30,
        "paan_money": 0,
        "lemon_money": 0,
        "paan_profit": 0,
        "lemon_profit": 0,
        # note_counts uses integer keys; we will convert when loading/saving
        "note_counts": {10: 0, 20: 0, 50: 0, 100: 0, 500: 0, 1000: 0, 5000: 0}
    }

def load_data():
    """Load saved data from JSON, converting note keys back to int."""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
            # Convert note_counts keys back to ints if present
            if "note_counts" in data and isinstance(data["note_counts"], dict):
                data["note_counts"] = {int(k): v for k, v in data["note_counts"].items()}
            # Ensure all expected keys exist (in case file is partial)
            base = default_data()
            base.update(data)
            return base
        except Exception as e:
            st.error(f"Error loading {DATA_FILE}: {e}")
            return default_data()
    else:
        return default_data()

def save_data():
    """Save current session state to JSON file."""
    data_to_save = {
        "paan_qty": st.session_state.paan_qty,
        "lemon_qty": st.session_state.lemon_qty,
        "paan_money": st.session_state.paan_money,
        "lemon_money": st.session_state.lemon_money,
        "paan_profit": st.session_state.paan_profit,
        "lemon_profit": st.session_state.lemon_profit,
        # JSON will convert integer keys to strings automatically
        "note_counts": st.session_state.note_counts
    }
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(data_to_save, f, indent=4)
    except Exception as e:
        st.error(f"Error saving data: {e}")

def reset_data():
    """Reset session_state values to defaults (does NOT auto-save)."""
    d = default_data()
    for k, v in d.items():
        st.session_state[k] = v

# --- Load saved data (or defaults) and initialize session_state ---
loaded = load_data()
for key, val in loaded.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Initialize confirmation flag for reset if absent
if "reset_confirm" not in st.session_state:
    st.session_state.reset_confirm = False

# --- Prices (constants) ---
PAAN_COST = 50
PAAN_SELL = 150
LEMON_COST = 50
LEMON_SELL = 150

# --- Actions (do NOT auto-save) ---
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

# --- UI ---
st.title("🍰 Bake Sale Dashboard")
st.markdown("Track items, sales, notes and manually Save / Reset when you want.")

# Items on sale
st.header("🧾 Items on Sale")
left, right = st.columns(2)

with left:
    st.subheader("🥬 Paan")
    st.write(f"Cost Price: Rs {PAAN_COST} — Sale Price: Rs {PAAN_SELL}")
    st.metric("Quantity Remaining", st.session_state.paan_qty)
    if st.button("Sell Paan", use_container_width=True):
        sell_paan()
        # don't auto-save; user will press Save when ready
        st.experimental_rerun()

with right:
    st.subheader("🍋 Lemon Soda")
    st.write(f"Cost Price: Rs {LEMON_COST} — Sale Price: Rs {LEMON_SELL}")
    st.metric("Quantity Remaining", st.session_state.lemon_qty)
    if st.button("Sell Lemon Soda", use_container_width=True):
        sell_lemon()
        st.experimental_rerun()

# Financials
paan_invested = 50 * PAAN_COST
lemon_invested = 30 * LEMON_COST
paan_received = st.session_state.paan_money
lemon_received = st.session_state.lemon_money
paan_profit = st.session_state.paan_profit
lemon_profit = st.session_state.lemon_profit
total_profit = paan_profit + lemon_profit

st.header("💰 Financial Summary")
c1, c2, c3 = st.columns(3)
c1.metric("Total Money Invested", f"Rs {paan_invested + lemon_invested}")
c2.metric("Total Money Received", f"Rs {paan_received + lemon_received}")
c3.metric("Total Profit", f"Rs {total_profit}")

st.markdown("#### 📊 Item Breakdown")
st.dataframe({
    "Item": ["Paan", "Lemon Soda"],
    "Price Invested (Rs)": [paan_invested, lemon_invested],
    "Money Received (Rs)": [paan_received, lemon_received],
    "Profit (Rs)": [paan_profit, lemon_profit],
}, use_container_width=True)

# Notes counter
st.header("💵 Cash Notes Counter")
notes = {10: "₨10", 20: "₨20", 50: "₨50", 100: "₨100", 500: "₨500", 1000: "₨1000", 5000: "₨5000"}

st.markdown("Manage note quantities (these changes are kept in session until you Save).")

for note, label in notes.items():
    cols = st.columns([2, 1, 1])
    with cols[0]:
        st.write(f"{label} : {st.session_state.note_counts[note]}")
    with cols[1]:
        if st.button(f"+{note}", key=f"add_{note}"):
            st.session_state.note_counts[note] += 1
            st.experimental_rerun()
    with cols[2]:
        if st.button(f"-{note}", key=f"sub_{note}"):
            if st.session_state.note_counts[note] > 0:
                st.session_state.note_counts[note] -= 1
            st.experimental_rerun()

total_money = sum(k * v for k, v in st.session_state.note_counts.items())
st.success(f"💵 Total Cash from Notes: Rs {total_money}")

# --- Save & Reset UI (manual) ---
st.markdown("---")
st.header("Save / Reset (manual)")

save_col, reset_col = st.columns([1, 1])

with save_col:
    if st.button("💾 Save Data to File", use_container_width=True):
        save_data()
        st.success("✅ Data saved to bakesale_data.json")

with reset_col:
    # First click toggles a confirmation flag shown to the user
    if not st.session_state.reset_confirm:
        if st.button("♻️ Reset All Data", use_container_width=True):
            st.session_state.reset_confirm = True
            st.experimental_rerun()
    else:
        st.warning("Are you sure? This will reset all values to defaults (50 / 30 etc).")
        confirm, cancel = st.columns(2)
        with confirm:
            if st.button("Confirm Reset", use_container_width=True):
                reset_data()
                st.session_state.reset_confirm = False
                # Optionally remove saved file so load starts fresh next run
                if os.path.exists(DATA_FILE):
                    try:
                        os.remove(DATA_FILE)
                    except Exception as e:
                        st.error(f"Could not remove {DATA_FILE}: {e}")
                st.success("✅ Data reset to defaults (not saved).")
                st.experimental_rerun()
        with cancel:
            if st.button("Cancel", use_container_width=True):
                st.session_state.reset_confirm = False
                st.experimental_rerun()

st.markdown("---")
st.caption("Note: Changes are kept in the session until you press 'Save Data'.")
