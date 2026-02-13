import streamlit as st
from rembg import remove
from PIL import Image, ImageEnhance, ImageFilter
import io
import numpy as np
import cv2

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Passport AI Pro",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. HIGH-END GRAPHICS & ANIMATION (CSS) ---
st.markdown("""
    <style>
        /* IMPORT FONT (Outfit - Modern Sans) */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&display=swap');

        /* ANIMATED AURORA BACKGROUND */
        .stApp {
            background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #141E30);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            font-family: 'Outfit', sans-serif;
            color: white;
        }
        
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* GLASSMORPHISM CARD CONTAINER */
        .glass-container {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 2rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            margin-bottom: 20px;
        }

        /* HERO TITLE GRADIENT TEXT */
        .hero-text {
            background: linear-gradient(to right, #00c6ff, #0072ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0;
        }

        /* CUSTOM UPLOAD WIDGET STYLING */
        .stFileUploader {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            border: 1px dashed rgba(255, 255, 255, 0.3);
        }

        /* PRIMARY ACTION BUTTONS */
        div.stButton > button {
            background: linear-gradient(92.88deg, #455EB5 9.16%, #5643CC 43.89%, #673FD7 64.72%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1.1rem;
            width: 100%;
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 4px 14px 0 rgba(0,118,255,0.39);
        }
        div.stButton > button:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 20px rgba(0,118,255,0.23);
        }

        /* BUY ME A COFFEE BUTTON */
        .coffee-btn {
            display: flex;
            align-items: center;
            justify-content: center;
            background: #FFDD00;
            color: #000;
            font-weight: 700;
            text-decoration: none;
            padding: 15px 30px;
            border-radius: 50px;
            box-shadow: 0 4px 15px rgba(255, 221, 0, 0.4);
            transition: all 0.3s ease;
            margin: 20px auto;
            width: fit-content;
        }
        .coffee-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(255, 221, 0, 0.6);
            color: black;
        }

        /* HIDE DEFAULT STREAMLIT ELEMENTS */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* TEXT OVERRIDES */
        h1, h2, h3, p, div, label { color: white !important; }
        .stSuccess { background-color: rgba(0, 255, 127, 0.1) !important; color: #00ff7f !important; }
        
    </style>
""", unsafe_allow_html=True)

# --- 3. PROCESSING LOGIC (Robust & Fast) ---
TARGET_W, TARGET_H = 630, 810
MAX_FILE_SIZE_KB = 250

def process_image(input_image):
    # 1. REMOVE BG
    buf = io.BytesIO()
    input_image.save(buf, format="PNG")
    subject_data = remove(buf.getvalue(), alpha_matting=True)
    foreground = Image.open(io.BytesIO(subject_data)).convert("RGBA")
    
    # 2. WHITE BG
    new_bg = Image.new("RGBA", foreground.size, "WHITE")
    new_bg.paste(foreground, (0, 0), foreground)
    final_rgb = new_bg.convert("RGB")
    
    # 3. DETECT & CROP
    opencv_img = np.array(final_rgb)
    opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        # ICAO Logic
        face_cx, face_cy = x + w//2, y + h//2
        head_h = h * 1.55 # Padding for hair
        req_h = int(head_h / 0.75)
        req_w = int(req_h * (TARGET_W / TARGET_H))
        
        c_x1 = face_cx - req_w // 2
        c_y1 = (face_cy - req_h // 2) - int(req_h * 0.1)
        c_x2, c_y2 = c_x1 + req_w, c_y1 + req_h
        
        # Pad and Crop
        final_rgb_padded = Image.new("RGB", (final_rgb.width + req_w*2, final_rgb.height + req_h*2), "WHITE")
        final_rgb_padded.paste(final_rgb, (req_w, req_h))
        final_rgb = final_rgb_padded.crop((c_x1+req_w, c_y1+req_h, c_x2+req_w, c_y2+req_h))

    # 4. RESIZE & COMPRESS
    final_rgb = final_rgb.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
    
    quality = 95
    while quality > 10:
        out_buf = io.BytesIO()
        final_rgb.save(out_buf, format="JPEG", quality=quality)
        if out_buf.tell() / 1024 < MAX_FILE_SIZE_KB:
            out_buf.seek(0)
            return out_buf
        quality -= 5
    return out_buf

# --- 4. UI LAYOUT ---

# Header
st.markdown('<h1 class="hero-text">Passport AI ✨</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; opacity: 0.8; margin-bottom: 30px;">Professional • Private • Instant</p>', unsafe_allow_html=True)

# Main Glass Card
st.markdown('<div class="glass-container">', unsafe_allow_html=True)

# Tabs
tab_up, tab_cam = st.tabs(["📤 Upload", "📸 Camera"])
img_file = None

with tab_up:
    uploaded = st.file_uploader("Drop your photo here", type=['jpg','png','jpeg'])
    if uploaded: img_file = uploaded

with tab_cam:
    # Camera Logic (Hidden by default)
    if "cam_active" not in st.session_state: st.session_state.cam_active = False
    
    if not st.session_state.cam_active:
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("🔵 Start Camera"):
                st.session_state.cam_active = True
                st.rerun()
    else:
        img_captured = st.camera_input("Center your face", label_visibility="collapsed")
        if img_captured: img_file = img_captured
        if st.button("❌ Close Camera"):
            st.session_state.cam_active = False
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Result Section
if img_file:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("### ✨ Magic Studio")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_file, caption="Original", use_container_width=True)
    
    with col2:
        # Process Button & Result
        process_btn = st.button("⚡ Generate Passport Photo")
        if process_btn:
            with st.spinner("Applying AI Magic..."):
                final_buffer = process_image(Image.open(img_file))
                
            st.success("Ready for Download!")
            st.image(final_buffer, caption="Passport Ready (630x810)", use_container_width=True)
            
            st.download_button(
                label="⬇️ Download HD Image",
                data=final_buffer,
                file_name="passport_photo.jpg",
                mime="image/jpeg"
            )
            
    st.markdown('</div>', unsafe_allow_html=True)

# --- 5. BUY ME A COFFEE (PREMIUM FOOTER) ---
st.markdown("""
    <br><br>
    <div style="text-align: center;">
        <p style="font-size: 1.2rem; margin-bottom: 10px;">Found this useful?</p>
        <a href="https://paypal.me/698789" target="_blank" class="coffee-btn">
            ☕ Buy me a Coffee
        </a>
        <p style="font-size: 0.8rem; opacity: 0.5; margin-top: 20px;">
            Secure Processing • No Data Saved • ICAO Compliant
        </p>
    </div>
""", unsafe_allow_html=True)