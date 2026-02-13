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
    </style>
""", unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if 'step' not in st.session_state: st.session_state.step = 1
if 'input_image' not in st.session_state: st.session_state.input_image = None
if 'processed_image' not in st.session_state: st.session_state.processed_image = None
if 'cam_active' not in st.session_state: st.session_state.cam_active = False

# --- 4. PROCESSING LOGIC (FIXED) ---
def process_photo(pil_img, std):
    # 1. Remove Background
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    subject = remove(buf.getvalue(), alpha_matting=True)
    foreground = Image.open(io.BytesIO(subject)).convert("RGBA")
    
    # 2. Create White Background
    bg = Image.new("RGBA", foreground.size, "WHITE")
    bg.paste(foreground, (0, 0), foreground)
    rgb_img = bg.convert("RGB")
    
    # 3. Detect Face
    opencv_img = np.array(rgb_img)
    opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY), 1.1, 5)
    
    # 4. RESTORED LOGIC: Calculate Crop & Safe Paste
    if len(faces) > 0:
        # Find largest face
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        
        # Center of face
        face_cx = x + w // 2
        face_cy = y + h // 2
        
        # Calculate Required Dimensions based on Head Size
        # (Head should be ~75% of height)
        head_padding = 1.55  
        req_h_pixels = int((h * head_padding) / 0.75)
        req_w_pixels = int(req_h_pixels * (std['w'] / std['h']))
        
        # Calculate Crop Box Coordinates (Relative to Original Image)
        crop_x1 = face_cx - req_w_pixels // 2
        # Shift crop slightly up so eyes are near top 1/3
        vertical_offset = int(req_h_pixels * 0.08) 
        crop_y1 = (face_cy - req_h_pixels // 2) - vertical_offset
        crop_x2 = crop_x1 + req_w_pixels
        crop_y2 = crop_y1 + req_h_pixels
        
        # SAFE CROP: Create a new blank canvas and paste the intersection
        # This handles cases where the head is too close to the edge
        canvas = Image.new("RGB", (req_w_pixels, req_h_pixels), "WHITE")
        
        # Calculate intersection with original image
        src_x1 = max(0, crop_x1)
        src_y1 = max(0, crop_y1)
        src_x2 = min(rgb_img.width, crop_x2)
        src_y2 = min(rgb_img.height, crop_y2)
        
        # Calculate where to paste on the canvas
        dst_x = max(0, -crop_x1)
        dst_y = max(0, -crop_y1)
        
        # Paste the valid region
        if src_x2 > src_x1 and src_y2 > src_y1:
            region = rgb_img.crop((src_x1, src_y1, src_x2, src_y2))
            canvas.paste(region, (dst_x, dst_y))
            
        rgb_img = canvas

    # 5. Final Resize to Target Standard (e.g. 630x810)
    final = rgb_img.resize((std['w'], std['h']), Image.Resampling.LANCZOS)
    
    # 6. Compress
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
    
    tab_up, tab_cam = st.tabs(["📤 Upload File", "📸 Take Selfie"])
    
    with tab_up:
        uploaded = st.file_uploader("Upload Photo", type=['jpg','png','jpeg'], label_visibility="collapsed")
        if uploaded:
            st.session_state.input_image = Image.open(uploaded)
            
            # Simple validation info
            curr_w, curr_h = st.session_state.input_image.size
            std = PHOTO_STANDARDS[selected]
            st.info(f"Original Size: {curr_w}x{curr_h} px. Target: {std['w']}x{std['h']} px.")
            
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

    st.markdown("### 💡 ICAO Compliance Guide")
    st.info("Head should be centered and occupy 70-80% of photo height.")

elif st.session_state.step == 2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ⚡ Step 2: Processing")
    with st.spinner("Aligning face and cropping correctly..."):
        time.sleep(1)
        buf, size = process_photo(st.session_state.input_image, PHOTO_STANDARDS[st.session_state.selected_std])
        st.session_state.processed_image, st.session_state.final_size = buf, size
        st.session_state.step = 3; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.step == 3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ✅ Step 3: Success!")
    
    st.image(st.session_state.processed_image, caption=f"Compliant Result: {st.session_state.final_size:.1f} KB", width=250)
    
    st.download_button("⬇️ Download High-Res Photo", st.session_state.processed_image, "passport.jpg", "image/jpeg")
    
    st.markdown(f'<br><a href="https://paypal.me/" target="_blank" class="paypal-btn">☕ Buy me a Coffee</a>', unsafe_allow_html=True)
    
    if st.button("🔄 New Photo"): 
        st.session_state.step = 1; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)