import streamlit as st
from rembg import remove
from PIL import Image
import io
import numpy as np
import cv2
import time
import pandas as pd
from specs import PHOTO_STANDARDS  # Import specs from the file above

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Global Passport Pro", page_icon="🛂", layout="centered")

# --- 2. PREMIUM CSS ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        .stApp { 
            background: linear-gradient(-45deg, #0f0c29, #1a1a2e, #16213e); 
            color: white; 
            font-family: 'Inter', sans-serif;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 20px;
        }
        .paypal-btn {
            background: #FFC439; color: black !important; padding: 12px 30px; 
            border-radius: 50px; text-decoration: none; font-weight: bold; display: inline-block;
        }
        h1, h3, p { text-align: center; }
        #MainMenu, footer, header {visibility: hidden;}
        /* Accessible Label Hiding */
        .stFileUploader label, .stCameraInput label { display: none; }
    </style>
""", unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if 'step' not in st.session_state: st.session_state.step = 1
if 'input_image' not in st.session_state: st.session_state.input_image = None
if 'processed_image' not in st.session_state: st.session_state.processed_image = None
if 'cam_active' not in st.session_state: st.session_state.cam_active = False

# --- 4. PROCESSING LOGIC ---
def process_photo(pil_img, std):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    subject = remove(buf.getvalue(), alpha_matting=True)
    foreground = Image.open(io.BytesIO(subject)).convert("RGBA")
    
    bg = Image.new("RGBA", foreground.size, "WHITE")
    bg.paste(foreground, (0, 0), foreground)
    rgb_img = bg.convert("RGB")
    
    # Auto-Crop Logic
    opencv_img = np.array(rgb_img)
    opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY), 1.1, 5)
    
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        face_cx, face_cy = x + w//2, y + h//2
        req_h = int((h * 1.55) / 0.75)
        req_w = int(req_h * (std['w'] / std['h']))
        canvas = Image.new("RGB", (rgb_img.width + req_w, rgb_img.height + req_h), "WHITE")
        canvas.paste(rgb_img, (req_w//2, req_h//2))
        rgb_img = canvas.crop((face_cx, face_cy - int(req_h*0.6), face_cx + req_w, face_cy + int(req_h*0.4)))

    final = rgb_img.resize((std['w'], std['h']), Image.Resampling.LANCZOS)
    
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
    st.markdown("### 📋 Step 1: Check Standards & Upload")
    
    # Show Standards Table
    df = pd.DataFrame(PHOTO_STANDARDS).T.reset_index()
    df = df.rename(columns={"index": "Standard", "mm": "Dim (mm)", "w": "Width", "h": "Height", "kb": "Max KB"})
    st.table(df[["Standard", "Dim (mm)", "Width", "Height", "Max KB"]])

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    selected = st.selectbox("Select Target Destination:", list(PHOTO_STANDARDS.keys()))
    st.session_state.selected_std = selected
    
    # Upload vs Camera Tabs
    tab_up, tab_cam = st.tabs(["📤 Upload File", "📸 Take Selfie"])
    
    with tab_up:
        uploaded = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], label_visibility="collapsed")
        if uploaded:
            st.session_state.input_image = Image.open(uploaded)
            # Basic Validation Check
            curr_w, curr_h = st.session_state.input_image.size
            std = PHOTO_STANDARDS[selected]
            if curr_w == std['w'] and curr_h == std['h']:
                st.success("✅ Photo matches dimensions!")
            else:
                st.warning(f"⚠️ Original: {curr_w}x{curr_h} px. Will be resized to {std['w']}x{std['h']} px.")
            
            if st.button("✨ Auto-Fix & Convert"):
                st.session_state.step = 2; st.rerun()

    with tab_cam:
        if not st.session_state.cam_active:
            if st.button("🔵 Open Camera"):
                st.session_state.cam_active = True; st.rerun()
        else:
            cam_snap = st.camera_input("Selfie", label_visibility="collapsed")
            if cam_snap:
                st.session_state.input_image = Image.open(cam_snap)
                st.session_state.step = 2; st.rerun()
            if st.button("❌ Close Camera"):
                st.session_state.cam_active = False; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ICAO COMPLIANCE GUIDE
    st.markdown("### 💡 ICAO Compliance Guide")
    st.info("""
    **To ensure your photo is accepted:**
    1. **Face:** Must be fully visible, looking straight at the camera.
    2. **Expression:** Neutral expression, mouth closed, eyes open.
    3. **Position:** Head should be centered and cover 70-80% of the photo height.
    4. **Lighting:** Even lighting, no shadows on face or background.
    """)

elif st.session_state.step == 2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ⚡ Step 2: Processing")
    with st.spinner("Removing background and aligning to international standards..."):
        time.sleep(1)
        buf, size = process_photo(st.session_state.input_image, PHOTO_STANDARDS[st.session_state.selected_std])
        st.session_state.processed_image, st.session_state.final_size = buf, size
        st.session_state.step = 3; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.step == 3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ✅ Step 3: Success!")
    
    # Display Result
    st.image(st.session_state.processed_image, caption=f"Compliant Result: {st.session_state.final_size:.1f} KB", width=250)
    
    st.download_button("⬇️ Download High-Res Photo", st.session_state.processed_image, "passport.jpg", "image/jpeg")
    
    st.markdown(f'<br><a href="https://paypal.me/698789" target="_blank" class="paypal-btn">☕ Buy me a Coffee</a>', unsafe_allow_html=True)
    
    if st.button("🔄 New Photo"): 
        st.session_state.step = 1; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)