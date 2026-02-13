import streamlit as st
from rembg import remove
from PIL import Image
import io
import numpy as np
import cv2
import time
import pandas as pd
from specs import PHOTO_STANDARDS  # Ensure specs.py is in the same folder

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Global Passport Pro", page_icon="🛂", layout="centered")

# --- 2. PREMIUM CSS ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&display=swap');
        .stApp { 
            background: linear-gradient(-45deg, #0f0c29, #1a1a2e, #16213e); 
            color: white; 
            font-family: 'Outfit', sans-serif;
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
        .stFileUploader label, .stCameraInput label { display: none; }
    </style>
""", unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if 'step' not in st.session_state: st.session_state.step = 1
if 'input_image' not in st.session_state: st.session_state.input_image = None
if 'processed_image' not in st.session_state: st.session_state.processed_image = None
if 'cam_active' not in st.session_state: st.session_state.cam_active = False
if 'target_quality' not in st.session_state: st.session_state.target_quality = "Standard (~250KB)"

# --- 4. PROCESSING LOGIC (UPDATED WITH QUALITY CONTROL) ---
def process_photo(pil_img, std, quality_mode):
    # 1. Remove Background
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    subject = remove(buf.getvalue(), alpha_matting=True)
    foreground = Image.open(io.BytesIO(subject)).convert("RGBA")
    
    # 2. White Background
    bg = Image.new("RGBA", foreground.size, "WHITE")
    bg.paste(foreground, (0, 0), foreground)
    rgb_img = bg.convert("RGB")
    
    # 3. Auto-Crop (Fixed Logic)
    opencv_img = np.array(rgb_img)
    opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY), 1.1, 5)
    
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        face_cx, face_cy = x + w // 2, y + h // 2
        
        # Head ~75% of height logic
        head_padding = 1.55  
        req_h_pixels = int((h * head_padding) / 0.75)
        req_w_pixels = int(req_h_pixels * (std['w'] / std['h']))
        
        crop_x1 = face_cx - req_w_pixels // 2
        vertical_offset = int(req_h_pixels * 0.08) 
        crop_y1 = (face_cy - req_h_pixels // 2) - vertical_offset
        crop_x2 = crop_x1 + req_w_pixels
        crop_y2 = crop_y1 + req_h_pixels
        
        # Safe Paste Canvas
        canvas = Image.new("RGB", (req_w_pixels, req_h_pixels), "WHITE")
        src_x1, src_y1 = max(0, crop_x1), max(0, crop_y1)
        src_x2, src_y2 = min(rgb_img.width, crop_x2), min(rgb_img.height, crop_y2)
        dst_x, dst_y = max(0, -crop_x1), max(0, -crop_y1)
        
        if src_x2 > src_x1 and src_y2 > src_y1:
            region = rgb_img.crop((src_x1, src_y1, src_x2, src_y2))
            canvas.paste(region, (dst_x, dst_y))
        rgb_img = canvas

    # 4. Resize
    final = rgb_img.resize((std['w'], std['h']), Image.Resampling.LANCZOS)
    
    # 5. Smart Compression based on User Selection
    out = io.BytesIO()
    
    if quality_mode == "Max Quality (Uncompressed)":
        # Save at 100% quality, 0 subsampling (Maximum Detail)
        final.save(out, format="JPEG", quality=100, subsampling=0)
        
    elif quality_mode == "Standard (~250 KB)":
        # Target roughly 250KB limit
        q = 95
        while q > 50:
            out = io.BytesIO()
            final.save(out, format="JPEG", quality=q)
            if out.tell() / 1024 < 250: break
            q -= 5
            
    elif quality_mode == "Strict Upload (< 100 KB)":
        # Aggressive compression for strict portals
        q = 90
        while q > 10:
            out = io.BytesIO()
            final.save(out, format="JPEG", quality=q)
            if out.tell() / 1024 < 100: break
            q -= 5

    out.seek(0)
    return out, out.tell() / 1024

# --- 5. UI FLOW ---
st.markdown("<h1>Global Passport Pro AI ✨</h1>", unsafe_allow_html=True)

if st.session_state.step == 1:
    st.markdown("### 📋 Step 1: Configure & Upload")
    
    # Standards Table
    df = pd.DataFrame(PHOTO_STANDARDS).T.reset_index()
    df = df.rename(columns={"index": "Standard", "mm": "Dim (mm)", "w": "Width", "h": "Height"})
    st.table(df[["Standard", "Dim (mm)", "Width", "Height"]])

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        selected = st.selectbox("1. Choose Country:", list(PHOTO_STANDARDS.keys()))
        st.session_state.selected_std = selected
    with col2:
        # NEW QUALITY SELECTOR
        q_mode = st.selectbox(
            "2. Select Output Size:", 
            ["Max Quality (Uncompressed)", "Standard (~250 KB)", "Strict Upload (< 100 KB)"],
            index=0,
            help="Max Quality gives the clearest image but larger file size."
        )
        st.session_state.target_quality = q_mode
    
    tab_up, tab_cam = st.tabs(["📤 Upload File", "📸 Take Selfie"])
    
    with tab_up:
        uploaded = st.file_uploader("Upload Photo", type=['jpg','png','jpeg'], label_visibility="collapsed")
        if uploaded:
            st.session_state.input_image = Image.open(uploaded)
            st.success("Photo Uploaded!")
            if st.button("✨ Process Photo"):
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

    st.markdown("### 💡 ICAO Compliance Guide")
    st.info("Head should be centered and occupy 70-80% of photo height.")

elif st.session_state.step == 2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ⚡ Step 2: Processing")
    
    with st.spinner("Applying AI background removal and quality settings..."):
        time.sleep(1)
        buf, size = process_photo(
            st.session_state.input_image, 
            PHOTO_STANDARDS[st.session_state.selected_std],
            st.session_state.target_quality  # Pass user preference
        )
        st.session_state.processed_image, st.session_state.final_size = buf, size
        st.session_state.step = 3; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.step == 3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ✅ Step 3: Success!")
    
    # Show precise file size
    st.image(st.session_state.processed_image, caption=f"Final Size: {st.session_state.final_size:.1f} KB", width=250)
    
    st.download_button("⬇️ Download Photo", st.session_state.processed_image, "passport.jpg", "image/jpeg")
    
    st.markdown(f'<br><a href="https://paypal.me/698789" target="_blank" class="paypal-btn">☕ Buy me a Coffee</a>', unsafe_allow_html=True)
    
    if st.button("🔄 Start Over"): 
        st.session_state.step = 1; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)