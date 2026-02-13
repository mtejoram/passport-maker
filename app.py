import streamlit as st
from rembg import remove
from PIL import Image
import io
import numpy as np
import cv2
import time

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Passport AI",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. SESSION STATE MANAGEMENT (The "Wizard" Logic) ---
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'input_image' not in st.session_state:
    st.session_state.input_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

def reset_app():
    st.session_state.step = 1
    st.session_state.input_image = None
    st.session_state.processed_image = None
    st.rerun()

# --- 3. HIGH-END STYLING ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        
        /* DARK AURORA BACKGROUND */
        .stApp {
            background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
            font-family: 'Inter', sans-serif;
            color: white;
        }
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* GLASS CARD */
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 30px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            text-align: center;
        }

        /* BUTTONS */
        div.stButton > button {
            background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
            color: #002d40;
            border: none;
            padding: 12px 24px;
            border-radius: 12px;
            font-weight: 700;
            font-size: 16px;
            width: 100%;
            transition: transform 0.2s;
        }
        div.stButton > button:hover {
            transform: scale(1.02);
            box-shadow: 0 0 20px rgba(0, 201, 255, 0.5);
        }

        /* SECONDARY BUTTON (Reset) */
        .reset-btn button {
            background: transparent !important;
            border: 1px solid rgba(255,255,255,0.3) !important;
            color: white !important;
        }

        /* PAYPAL BUTTON */
        .paypal-btn {
            display: inline-block;
            background: #FFC439;
            color: #000;
            font-weight: bold;
            text-decoration: none;
            padding: 15px 40px;
            border-radius: 50px;
            margin-top: 20px;
            box-shadow: 0 4px 15px rgba(255, 196, 57, 0.4);
            transition: all 0.3s;
        }
        .paypal-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(255, 196, 57, 0.6);
            color: black;
        }

        /* HIDE STREAMLIT UI */
        #MainMenu, header, footer {visibility: hidden;}
        
        h1 { font-size: 2.5rem !important; margin-bottom: 0 !important; }
        p { color: #ccc !important; }
    </style>
""", unsafe_allow_html=True)

# --- 4. PROCESSING LOGIC ---
TARGET_W, TARGET_H = 630, 810
MAX_FILE_SIZE_KB = 250

def process_image(pil_img):
    # Convert to bytes
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    
    # Remove BG
    subject = remove(buf.getvalue(), alpha_matting=True)
    foreground = Image.open(io.BytesIO(subject)).convert("RGBA")
    
    # White BG
    bg = Image.new("RGBA", foreground.size, "WHITE")
    bg.paste(foreground, (0, 0), foreground)
    rgb_img = bg.convert("RGB")
    
    # Resize
    final_img = rgb_img.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
    
    # Compress
    quality = 95
    out_buf = io.BytesIO()
    while quality > 10:
        out_buf = io.BytesIO()
        final_img.save(out_buf, format="JPEG", quality=quality)
        if out_buf.tell() / 1024 < MAX_FILE_SIZE_KB:
            break
        quality -= 5
    out_buf.seek(0)
    return out_buf

# --- 5. APP FLOW (PAGINATION) ---

# HEADER
st.markdown("<h1 style='text-align: center;'>Passport AI ✨</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center;'>Step {st.session_state.step} of 3</p>", unsafe_allow_html=True)

# PROGRESS BAR
st.progress(st.session_state.step / 3)

# CONTAINER
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

# --- STEP 1: UPLOAD ---
if st.session_state.step == 1:
    st.markdown("### 📸 Select Your Photo")
    
    tab_up, tab_cam = st.tabs(["Upload File", "Use Camera"])
    
    with tab_up:
        uploaded = st.file_uploader("", type=['jpg','png','jpeg'], label_visibility="collapsed")
        if uploaded:
            st.session_state.input_image = Image.open(uploaded)
            st.session_state.step = 2
            st.rerun()

    with tab_cam:
        if "cam_active" not in st.session_state: st.session_state.cam_active = False
        
        if not st.session_state.cam_active:
            if st.button("Start Camera"):
                st.session_state.cam_active = True
                st.rerun()
        else:
            cam_snap = st.camera_input("Center Face", label_visibility="collapsed")
            if cam_snap:
                st.session_state.input_image = Image.open(cam_snap)
                st.session_state.step = 2
                st.rerun()

# --- STEP 2: PROCESSING ---
elif st.session_state.step == 2:
    st.markdown("### ⚡ Processing...")
    
    # Show thumbnail
    st.image(st.session_state.input_image, width=150, caption="Original")
    
    with st.spinner("Removing background & resizing..."):
        # Simulate delay for UX (feels more "powerful")
        time.sleep(1) 
        result = process_image(st.session_state.input_image)
        st.session_state.processed_image = result
        st.session_state.step = 3
        st.rerun()

# --- STEP 3: RESULT & DOWNLOAD ---
elif st.session_state.step == 3:
    st.markdown("### ✅ Success!")
    
    # Show Result
    st.image(st.session_state.processed_image, caption="Passport Ready (630x810)", width=200)
    
    # DOWNLOAD BUTTON
    st.download_button(
        label="⬇️ Download Photo",
        data=st.session_state.processed_image,
        file_name="passport_photo.jpg",
        mime="image/jpeg"
    )
    
    # PAYPAL LINK (Prominent)
    st.markdown("""
        <br>
        <p style="font-size: 0.9rem;">Did this save you time?</p>
        <a href="https://paypal.me/698789" target="_blank" class="paypal-btn">
            ☕ Buy me a Coffee
        </a>
    """, unsafe_allow_html=True)
    
    # START OVER
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("🔄 Start Over", type="secondary"):
        reset_app()

st.markdown('</div>', unsafe_allow_html=True)