import streamlit as st
import random

# Page config
st.set_page_config(page_title="Flappy Bird", layout="centered")

# Initialize session state
if 'game_started' not in st.session_state:
    st.session_state.game_started = False
    st.session_state.bird_y = 250
    st.session_state.bird_velocity = 0
    st.session_state.pipes = []
    st.session_state.score = 0
    st.session_state.game_over = False
    st.session_state.frame_count = 0

# Game constants
GRAVITY = 0.6
JUMP_STRENGTH = -10
BIRD_X = 100
PIPE_WIDTH = 60
PIPE_GAP = 180
PIPE_SPEED = 3
SPAWN_INTERVAL = 90

def reset_game():
    st.session_state.game_started = True
    st.session_state.bird_y = 250
    st.session_state.bird_velocity = 0
    st.session_state.pipes = []
    st.session_state.score = 0
    st.session_state.game_over = False
    st.session_state.frame_count = 0

def jump():
    if not st.session_state.game_over:
        st.session_state.bird_velocity = JUMP_STRENGTH
        if not st.session_state.game_started:
            st.session_state.game_started = True

def update_game():
    if not st.session_state.game_started or st.session_state.game_over:
        return
    
    # Update bird
    st.session_state.bird_velocity += GRAVITY
    st.session_state.bird_y += st.session_state.bird_velocity
    
    # Check ceiling and floor collision
    if st.session_state.bird_y < 0 or st.session_state.bird_y > 550:
        st.session_state.game_over = True
        return
    
    # Spawn pipes
    st.session_state.frame_count += 1
    if st.session_state.frame_count % SPAWN_INTERVAL == 0:
        gap_y = random.randint(150, 350)
        st.session_state.pipes.append({'x': 400, 'gap_y': gap_y})
    
    # Update pipes
    pipes_to_remove = []
    for pipe in st.session_state.pipes:
        pipe['x'] -= PIPE_SPEED
        
        # Check if bird passed pipe
        if pipe['x'] == BIRD_X - PIPE_WIDTH:
            st.session_state.score += 1
        
        # Remove off-screen pipes
        if pipe['x'] < -PIPE_WIDTH:
            pipes_to_remove.append(pipe)
        
        # Check collision
        if BIRD_X + 30 > pipe['x'] and BIRD_X < pipe['x'] + PIPE_WIDTH:
            if st.session_state.bird_y < pipe['gap_y'] - PIPE_GAP//2 or \
               st.session_state.bird_y + 30 > pipe['gap_y'] + PIPE_GAP//2:
                st.session_state.game_over = True
    
    # Remove old pipes
    for pipe in pipes_to_remove:
        st.session_state.pipes.remove(pipe)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to bottom, #4ec0ca 0%, #87ceeb 100%);
    }
    .game-container {
        background: linear-gradient(to bottom, #4ec0ca 0%, #87ceeb 70%, #ded895 70%, #ded895 100%);
        border: 4px solid #000;
        border-radius: 10px;
        position: relative;
        margin: 20px auto;
        overflow: hidden;
    }
    .bird {
        background: #ffd700;
        border: 2px solid #ff8c00;
        border-radius: 50%;
        position: absolute;
    }
    .pipe {
        background: #228b22;
        border: 3px solid #006400;
        position: absolute;
    }
    .score {
        font-size: 48px;
        font-weight: bold;
        color: white;
        text-shadow: 3px 3px 0px #000;
        text-align: center;
        margin-top: 20px;
    }
    .game-over-text {
        font-size: 64px;
        font-weight: bold;
        color: red;
        text-shadow: 4px 4px 0px #000;
        text-align: center;
        position: absolute;
        top: 40%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 100;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🐦 Flappy Bird")

# Game canvas
canvas_html = f"""
<div class="game-container" style="width: 400px; height: 600px;">
    <div class="score" style="position: absolute; top: 20px; left: 50%; transform: translateX(-50%); z-index: 10;">
        {st.session_state.score}
    </div>
"""

# Add bird
canvas_html += f"""
    <div class="bird" style="left: {BIRD_X}px; top: {st.session_state.bird_y}px; width: 30px; height: 30px;"></div>
"""

# Add pipes
for pipe in st.session_state.pipes:
    top_height = pipe['gap_y'] - PIPE_GAP // 2
    bottom_top = pipe['gap_y'] + PIPE_GAP // 2
    bottom_height = 600 - bottom_top
    
    canvas_html += f"""
    <div class="pipe" style="left: {pipe['x']}px; top: 0px; width: {PIPE_WIDTH}px; height: {top_height}px;"></div>
    <div class="pipe" style="left: {pipe['x']}px; top: {bottom_top}px; width: {PIPE_WIDTH}px; height: {bottom_height}px;"></div>
    """

# Add game over text
if st.session_state.game_over:
    canvas_html += """
    <div class="game-over-text">GAME OVER</div>
    """

canvas_html += "</div>"

st.markdown(canvas_html, unsafe_allow_html=True)

# Controls
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    if st.button("🚀 JUMP", use_container_width=True, type="primary"):
        jump()
        update_game()
        st.rerun()

if st.session_state.game_over:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Play Again", use_container_width=True):
            reset_game()
            st.rerun()

# Auto-update game
if st.session_state.game_started and not st.session_state.game_over:
    update_game()
    st.rerun()

# Instructions
with st.expander("📖 How to Play"):
    st.write("""
    - Click the **JUMP** button to make the bird fly
    - Avoid hitting the pipes or the ground
    - Each pipe you pass gives you 1 point
    - Try to get the highest score!
    """)

st.info("💡 Tip: Keep clicking JUMP to keep the bird in the air!")
