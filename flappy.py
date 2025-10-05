
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import random

# --- Streamlit Setup ---
st.set_page_config(page_title="Flappy Bird in Streamlit", layout="centered")
st.title("🐦 Flappy Bird in Streamlit")

# --- Session State ---
if "bird_y" not in st.session_state:
    st.session_state.bird_y = 0.5
    st.session_state.velocity = 0.0
    st.session_state.pipes = []
    st.session_state.score = 0
    st.session_state.alive = True
    st.session_state.started = False

# --- Controls ---
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("FLAP"):
        st.session_state.velocity = -0.03
        st.session_state.started = True
with col2:
    if st.button("RESET"):
        st.session_state.bird_y = 0.5
        st.session_state.velocity = 0.0
        st.session_state.pipes = []
        st.session_state.score = 0
        st.session_state.alive = True
        st.session_state.started = False

game_area = st.empty()
score_area = st.empty()

# --- Game Loop ---
if st.session_state.alive and st.session_state.started:
    # Physics
    st.session_state.velocity += 0.002
    st.session_state.bird_y += st.session_state.velocity

    # Add new pipes
    if len(st.session_state.pipes) == 0 or st.session_state.pipes[-1][0] < 0.6:
        gap_y = random.uniform(0.2, 0.8)
        st.session_state.pipes.append([1.0, gap_y])

    # Move pipes
    for pipe in st.session_state.pipes:
        pipe[0] -= 0.01

    # Remove old pipes
    st.session_state.pipes = [p for p in st.session_state.pipes if p[0] > -0.1]

    # Collision detection
    for x, gap_y in st.session_state.pipes:
        if abs(x - 0.2) < 0.05:  # bird near pipe
            if not (gap_y - 0.2 < st.session_state.bird_y < gap_y + 0.2):
                st.session_state.alive = False
        if x < 0.2 and not hasattr(st.session_state, "scored_" + str(id((x, gap_y)))):
            st.session_state.score += 1
            setattr(st.session_state, "scored_" + str(id((x, gap_y))), True)

    # Floor/Ceiling
    if st.session_state.bird_y < 0 or st.session_state.bird_y > 1:
        st.session_state.alive = False

# --- Render Frame ---
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Bird
ax.add_patch(plt.Circle((0.2, st.session_state.bird_y), 0.03, color="yellow"))

# Pipes
for x, gap_y in st.session_state.pipes:
    ax.add_patch(plt.Rectangle((x - 0.05, 0), 0.1, gap_y - 0.2, color="green"))
    ax.add_patch(plt.Rectangle((x - 0.05, gap_y + 0.2), 0.1, 1, color="green"))

ax.axis("off")
game_area.pyplot(fig)

if st.session_state.alive:
    score_area.subheader(f"Score: {st.session_state.score}")
else:
    score_area.error(f"💀 Game Over! Final Score: {st.session_state.score}")

# --- Auto-refresh ---
time.sleep(0.03)
st.experimental_rerun()
