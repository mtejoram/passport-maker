import streamlit as st
from rembg import remove
from PIL import Image, ImageFilter
import io
import numpy as np
import cv2
import time
import pandas as pd
from specs import PHOTO_STANDARDS

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Global Passport Pro", page_icon="🛂", layout="centered")

# --- 2. CSS STYLING ---
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
        .report-box {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            padding: 15px;
            margin-top: 10px;
            font-size: 0.9rem;
        }
        .success-box { border-left: 5px solid #00ff7f; }
        .fail-box { border-left: 5px solid #ff4b4b; }
        
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
if 'file_size_kb' not in st.session_state: st.session_state.file_size_kb = 0
if 'cam_active' not in st.session_state: st.session_state.cam_active = False
if 'validation_result' not in st.session_state: st.session_state.validation_result = None

# --- 4. VALIDATION ENGINE ---
def analyze_image(pil_img, size_kb, std_key):
    std = PHOTO_STANDARDS[std_key]
    w, h = pil_img.size
    
    # Check 1: Dimensions
    dim_check = (w == std['w'] and h == std['h'])
    
    # Check 2: File Size (Must be under limit)
    size_check = (size_kb <= std['kb'])
    
    # Check 3: ICAO Face Geometry (Head 70-80%)
    opencv_img = np.array(pil_img.convert("RGB"))
    opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    icao_check = False
    face_ratio = 0
    face_msg = "No Face Detected"
    
    if len(faces) > 0:
        # Get largest face
        _, _, _, fh = max(faces, key=lambda b: b[2] * b[3])
        face_ratio = (fh / h) * 100
        
        # ICAO Standard: Head should be between 70% and 80% of image height
        # We allow a slightly wider range (60-85%) for "Pass" to be lenient
        if 60 <= face_ratio <= 85:
            icao_check = True
            face_msg = f"Good Head Size ({face_ratio:.1f}%)"
        else:
            face_msg = f"Head too small/large ({face_ratio:.1f}%)"
    
    is_compliant = dim_check and size_check and icao_check
    
    return {
        "is_compliant": is_compliant,
        "dim_ok": dim_check,
        "size_ok": size_check,
        "icao_ok": icao_check,
        "face_msg": face_msg,
        "current_w": w,
        "current_h": h,
        "current_kb": size_kb
    }

# --- 5. PROCESSING LOGIC ---
def process_photo(pil_img, std, quality_mode):
    # 1. Remove Background
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    subject = remove(buf.getvalue(), alpha_matting=True, alpha_matting_foreground_threshold=240)
    foreground = Image.open(io.BytesIO(subject)).convert("RGBA")
    
    # 2. Smooth Edges
    r, g, b, a = foreground.split()
    a = a.filter(ImageFilter.GaussianBlur(radius=0.5))
    foreground = Image.merge("RGBA", (r, g, b, a))

    # 3. White BG
    bg = Image.new("RGBA", foreground.size, "WHITE")
    bg.paste(foreground, (0, 0), foreground)
    rgb_img = bg.convert("RGB")
    
    # 4. Alpha-Aware Crop
    alpha_np = np.array(foreground)[:, :, 3]
    rows = np.where(np.max(alpha_np, axis=1) > 0)[0]
    
    if len(rows) > 0:
        top_y, bottom_y = rows[0], rows[-1]
        opencv_img = np.array(rgb_img)
        opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY), 1.1, 5)
        
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            cx, chin_y = x + w // 2, y + h
        else:
            cx, chin_y = rgb_img.width // 2, bottom_y - (bottom_y - top_y) // 4

        head_h = chin_y - top_y
        if head_h < 1: head_h = rgb_img.height // 2
            
        req_h = int(head_h / 0.75) 
        req_w = int(req_h * (std['w'] / std['h']))
        
        crop_y1 = top_y - int(req_h * 0.125)
        crop_y2 = crop_y1 + req_h
        crop_x1 = cx - req_w // 2
        crop_x2 = crop_x1 + req_w
        
        canvas = Image.new("RGB", (req_w, req_h), "WHITE")
        src_x1, src_y1 = max(0, crop_x1), max(0, crop_y1)
        src_x2, src_y2 = min(rgb_img.width, crop_x2), min(rgb_img.height, crop_y2)
        dst_x, dst_y = max(0, -crop_x1), max(0, -crop_y1)
        
        if src_x2 > src_x1 and src_y2 > src_y1:
            region = rgb_img.crop((src_x1, src_y1, src_x2, src_y2))
            canvas.paste(region, (dst_x, dst_y))
        rgb_img = canvas

    # 5. Resize & Compress
    final = rgb_img.resize((std['w'], std['h']), Image.Resampling.LANCZOS)
    
    out = io.BytesIO()
    if quality_mode == "Max Quality (Uncompressed)":
        final.save(out, format="JPEG", quality=100, subsampling=0)
    elif quality_mode == "Standard (~250 KB)":
        q = 95
        while q > 50:
            out = io.BytesIO()
            final.save(out, format="JPEG", quality=q)
            if out.tell() / 1024 < 250: break
            q -= 5
    elif quality_mode == "Strict Upload (< 100 KB)":
        q = 90
        while q > 10:
            out = io.BytesIO()
            final.save(out, format="JPEG", quality=q)
            if out.tell() / 1024 < 100: break
            q -= 5

    out.seek(0)
    return out, out.tell() / 1024

# --- 6. UI FLOW ---
st.markdown("<h1>Global Passport Pro AI ✨</h1>", unsafe_allow_html=True)

if st.session_state.step == 1:
    st.markdown("### 📋 Step 1: Standards & Upload")
    
    # Standards Table
    df = pd.DataFrame(PHOTO_STANDARDS).T.reset_index()
    df = df.rename(columns={"index": "Country", "mm": "Size (mm)", "w": "Px W", "h": "Px H", "kb": "Max KB"})
    st.dataframe(df[["Country", "Size (mm)", "Px W", "Px H", "Max KB"]], hide_index=True, use_container_width=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        selected = st.selectbox("Select Country:", list(PHOTO_STANDARDS.keys()))
        st.session_state.selected_std = selected
    with col_b:
        q_mode = st.selectbox("Output Quality:", ["Standard (~250 KB)", "Max Quality (Uncompressed)", "Strict Upload (< 100 KB)"])
        st.session_state.target_quality = q_mode
    
    tab_up, tab_cam = st.tabs(["📤 Upload File", "📸 Take Selfie"])
    
    with tab_up:
        uploaded = st.file_uploader("Upload", type=['jpg','png','jpeg'], label_visibility="collapsed")
        if uploaded:
            st.session_state.input_image = Image.open(uploaded)
            st.session_state.file_size_kb = uploaded.size / 1024
            # Run Analysis
            st.session_state.validation_result = analyze_image(st.session_state.input_image, st.session_state.file_size_kb, selected)
            st.rerun()

    with tab_cam:
        if not st.session_state.cam_active:
            if st.button("🔵 Open Camera"): st.session_state.cam_active = True; st.rerun()
        else:
            cam_snap = st.camera_input("Selfie", label_visibility="collapsed")
            if cam_snap:
                st.session_state.input_image = Image.open(cam_snap)
                st.session_state.file_size_kb = cam_snap.size / 1024
                # Run Analysis
                st.session_state.validation_result = analyze_image(st.session_state.input_image, st.session_state.file_size_kb, selected)
                st.rerun()
            if st.button("❌ Close Camera"): st.session_state.cam_active = False; st.rerun()
    
    # --- VALIDATION REPORT SECTION ---
    if st.session_state.input_image:
        res = st.session_state.validation_result
        std = PHOTO_STANDARDS[selected]
        
        st.divider()
        st.markdown("### 🔍 Photo Analysis Report")
        
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.image(st.session_state.input_image, caption="Your Photo", width=150)
        
        with col2:
            # Build Metrics Table
            metrics = {
                "Metric": ["Dimensions (Px)", "File Size (KB)", "Face Position (ICAO)"],
                "Your Photo": [
                    f"{res['current_w']} x {res['current_h']}", 
                    f"{res['current_kb']:.1f} KB",
                    res['face_msg']
                ],
                "Required": [
                    f"{std['w']} x {std['h']}", 
                    f"< {std['kb']} KB",
                    "Face 70-80% of Height"
                ],
                "Status": [
                    "✅ Pass" if res['dim_ok'] else "❌ Fail",
                    "✅ Pass" if res['size_ok'] else "❌ Fail",
                    "✅ Pass" if res['icao_ok'] else "⚠️ Check"
                ]
            }
            st.table(pd.DataFrame(metrics))

        # DECISION BLOCK
        if res['is_compliant']:
            st.markdown("""
                <div class="glass-card success-box" style="text-align: center;">
                    <h2 style="color: #00ff7f; margin:0;">🎉 Congratulations!</h2>
                    <p>Your photo already meets all standards for <b>{}</b>.</p>
                </div>
            """.format(selected.split(" - ")[0]), unsafe_allow_html=True)
            
            # Allow Direct Download Buffer creation
            buf = io.BytesIO()
            st.session_state.input_image.save(buf, format="JPEG")
            st.download_button("⬇️ Download As-Is", buf.getvalue(), "compliant_photo.jpg", "image/jpeg")
            
            st.write("--- OR ---")
            if st.button("✨ Re-Process Anyway (Better Background)"):
                st.session_state.step = 2; st.rerun()
                
        else:
            st.markdown("""
                <div class="glass-card fail-box">
                    <h4 style="color: #ff4b4b; margin:0;">⚠️ Issues Detected</h4>
                    <p>This photo does not meet government standards yet. Use our AI tool to fix it automatically.</p>
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("✨ Auto-Fix & Convert"):
                st.session_state.step = 2; st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# STEP 2 & 3 (Same Processing Flow)
elif st.session_state.step == 2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ⚡ Step 2: Processing")
    with st.spinner("Correcting geometry, background, and lighting..."):
        time.sleep(1)
        buf, size = process_photo(
            st.session_state.input_image, 
            PHOTO_STANDARDS[st.session_state.selected_std],
            st.session_state.target_quality
        )
        st.session_state.processed_image, st.session_state.final_size = buf, size
        st.session_state.step = 3; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.step == 3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ✅ Step 3: Success!")
    st.image(st.session_state.processed_image, caption=f"Final: {st.session_state.final_size:.1f} KB", width=250)
    st.download_button("⬇️ Download Photo", st.session_state.processed_image, "passport.jpg", "image/jpeg")
    st.markdown(f'<br><a href="https://paypal.me/698789" target="_blank" class="paypal-btn">☕ Buy me a Coffee</a>', unsafe_allow_html=True)
    if st.button("🔄 Start Over"): st.session_state.step = 1; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)