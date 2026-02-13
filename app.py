import streamlit as st
from rembg import remove
from PIL import Image
import io
import numpy as np
import cv2
import time
import pandas as pd
from specs import PHOTO_STANDARDS

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Global Passport Pro", page_icon="🛂", layout="centered")

# --- 2. ENHANCED CSS ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&display=swap');
        .stApp { background: linear-gradient(-45deg, #0f0c29, #1a1a2e, #16213e); font-family: 'Outfit', sans-serif; color: white; }
        .glass-card { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); border-radius: 20px; padding: 25px; border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 20px; }
        .status-badge { padding: 4px 12px; border-radius: 50px; font-size: 0.8rem; font-weight: bold; }
        .badge-pass { background: #00ff7f; color: #000; }
        .badge-fail { background: #ff4b4b; color: #fff; }
        .paypal-btn { background: #FFC439; color: black !important; padding: 12px 30px; border-radius: 50px; text-decoration: none; font-weight: bold; display: inline-block; }
        h1, h3, p { text-align: center; }
        #MainMenu, footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if 'step' not in st.session_state: st.session_state.step = 1
if 'input_image' not in st.session_state: st.session_state.input_image = None
if 'processed_image' not in st.session_state: st.session_state.processed_image = None
if 'validation_report' not in st.session_state: st.session_state.validation_report = {}

def reset_app():
    st.session_state.step = 1
    st.session_state.input_image = None
    st.session_state.processed_image = None
    st.session_state.validation_report = {}
    st.rerun()

# --- 4. SMART VALIDATION LOGIC ---
def analyze_photo_compliance(pil_img, std_key):
    std = PHOTO_STANDARDS[std_key]
    w, h = pil_img.size
    
    # 1. Face Detection
    opencv_img = np.array(pil_img.convert("RGB"))
    opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    report = {
        "Dimensions": "✅ Match" if (w == std['w'] and h == std['h']) else "❌ Wrong Size",
        "Face Detected": "✅ Yes" if len(faces) > 0 else "❌ No Face Found",
        "Background": "🟡 Check Needed" # AI check for white background
    }
    
    # Check face geometry if face exists
    if len(faces) > 0:
        fx, fy, fw, fh = faces[0]
        face_ratio = (fh / h) * 100
        report["Face Ratio"] = f"✅ {face_ratio:.0f}%" if (70 <= face_ratio <= 80) else f"❌ {face_ratio:.0f}% (Needs 70-80%)"
    
    return report

def process_photo(pil_img, std):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    subject = remove(buf.getvalue(), alpha_matting=True)
    foreground = Image.open(io.BytesIO(subject)).convert("RGBA")
    
    bg = Image.new("RGBA", foreground.size, "WHITE")
    bg.paste(foreground, (0, 0), foreground)
    rgb_img = bg.convert("RGB")
    
    # Advanced ICAO Face Centering
    opencv_img = np.array(rgb_img)
    opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY), 1.1, 5)
    
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        face_cx, face_cy = x + w//2, y + h//2
        req_h = int((h * 1.55) / 0.75)
        req_w = int(req_h * (std['w'] / std['h']))
        canvas = Image.new("RGB", (rgb_img.width + req_w, rgb_img.height + req_h), "WHITE")
        canvas.paste(rgb_img, (req_w//2, req_h//2))
        rgb_img = canvas.crop((face_cx, face_cy - int(req_h*0.6), face_cx + req_w, face_cy + int(req_h*0.4)))

    final = rgb_img.resize((std['w'], std['h']), Image.Resampling.LANCZOS)
    
    # Save with KB target
    quality = 95
    while quality > 10:
        out = io.BytesIO()
        final.save(out, format="JPEG", quality=quality)
        if out.tell() / 1024 < std['kb']: break
        quality -= 5
    out.seek(0)
    return out, out.tell() / 1024

# --- 5. UI FLOW ---
st.markdown("<h1>Global Passport Pro AI ✨</h1>", unsafe_allow_html=True)

if st.session_state.step == 1:
    st.markdown("### 📋 Step 1: Destination & Source")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    selected = st.selectbox("I need a photo for:", list(PHOTO_STANDARDS.keys()))
    st.session_state.selected_std = selected
    
    tab_up, tab_cam = st.tabs(["📤 Upload Photo", "📸 Take Selfie"])
    img = None
    with tab_up:
        up = st.file_uploader("", type=['jpg','png','jpeg'], label_visibility="collapsed")
        if up: img = Image.open(up)
    with tab_cam:
        snap = st.camera_input("Center yourself", label_visibility="collapsed")
        if snap: img = Image.open(snap)
        
    if img:
        st.session_state.input_image = img
        st.session_state.validation_report = analyze_photo_compliance(img, selected)
        st.session_state.step = 2; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.step == 2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Step 2: Compliance Report")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(st.session_state.input_image, use_container_width=True, caption="Source")
    with col_b:
        st.write("**Validation Checks:**")
        for check, status in st.session_state.validation_report.items():
            st.write(f"{status} **{check}**")
    
    st.divider()
    if st.button("✨ Auto-Fix Everything"):
        with st.spinner("AI is correcting geometry and background..."):
            buf, size = process_photo(st.session_state.input_image, PHOTO_STANDARDS[st.session_state.selected_std])
            st.session_state.processed_image, st.session_state.final_size = buf, size
            st.session_state.step = 3; st.rerun()
    
    if st.button("⬅️ Try Different Photo", type="secondary"): reset_app()
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.step == 3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ✅ Step 3: Result")
    st.image(st.session_state.processed_image, caption=f"Validated: {st.session_state.final_size:.1f} KB", width=250)
    st.download_button("⬇️ Download High-Res JPG", st.session_state.processed_image, "passport.jpg", "image/jpeg")
    st.markdown(f'<br><a href="https://paypal.me/698789" target="_blank" class="paypal-btn">☕ Buy me a Coffee</a>', unsafe_allow_html=True)
    if st.button("🔄 New Photo"): reset_app()
    st.markdown('</div>', unsafe_allow_html=True)