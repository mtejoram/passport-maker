import streamlit as st
from rembg import remove
from PIL import Image, ImageFilter, ImageOps
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
        .success-box { border-left: 5px solid #00ff7f; background: rgba(0, 255, 127, 0.1); padding: 15px; border-radius: 5px; }
        .fail-box { border-left: 5px solid #ff4b4b; background: rgba(255, 75, 75, 0.1); padding: 15px; border-radius: 5px; }
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

# --- 4. SAFETY UTILS ---
def resize_if_huge(img, max_dim=1024):
    """Resizes image before processing to prevent memory crashes."""
    w, h = img.size
    if w > max_dim or h > max_dim:
        ratio = min(max_dim/w, max_dim/h)
        return img.resize((int(w*ratio), int(h*ratio)), Image.Resampling.LANCZOS)
    return img

# --- 5. VALIDATION ENGINE ---
def analyze_image(pil_img, size_kb, std_key):
    std = PHOTO_STANDARDS[std_key]
    w, h = pil_img.size
    
    dim_check = (w == std['w'] and h == std['h'])
    size_check = (size_kb <= std['kb'])
    
    # Analyze Face
    opencv_img = np.array(pil_img.convert("RGB"))
    opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    icao_check = False
    face_msg = "No Face Detected"
    
    if len(faces) > 0:
        _, _, _, fh = max(faces, key=lambda b: b[2] * b[3])
        face_ratio = (fh / h) * 100
        if 60 <= face_ratio <= 85:
            icao_check = True
            face_msg = f"Good Head Size ({face_ratio:.1f}%)"
        else:
            face_msg = f"Head Size ({face_ratio:.1f}%)"
    
    return {
        "is_compliant": dim_check and size_check and icao_check,
        "dim_ok": dim_check,
        "size_ok": size_check,
        "icao_ok": icao_check,
        "face_msg": face_msg,
        "current_w": w, "current_h": h, "current_kb": size_kb
    }

# --- 6. PROCESSING LOGIC ---
@st.cache_data(show_spinner=False)
def process_photo(pil_img, std, quality_mode):
    try:
        # 0. Safety Resize
        work_img = resize_if_huge(pil_img)

        # 1. Remove Background
        buf = io.BytesIO()
        work_img.save(buf, format="PNG")
        
        # 'alpha_matting' is heavy, so we wrap it
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
        
        # 4. Smart Crop
        alpha_np = np.array(foreground)[:, :, 3]
        rows = np.where(np.max(alpha_np, axis=1) > 0)[0]
        
        if len(rows) > 0:
            top_y, bottom_y = rows[0], rows[-1]
            # Face Detect
            cv_img = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.1, 5)
            
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                cx = x + w // 2
                chin_y = y + h
            else:
                cx = rgb_img.width // 2
                chin_y = bottom_y - (bottom_y - top_y) // 4

            head_h = chin_y - top_y
            if head_h < 1: head_h = rgb_img.height // 2
                
            req_h = int(head_h / 0.75) 
            req_w = int(req_h * (std['w'] / std['h']))
            
            crop_y1 = top_y - int(req_h * 0.125)
            crop_x1 = cx - req_w // 2
            
            # Safe Crop Canvas
            canvas = Image.new("RGB", (req_w, req_h), "WHITE")
            src_x1, src_y1 = max(0, crop_x1), max(0, crop_y1)
            src_x2, src_y2 = min(rgb_img.width, crop_x1+req_w), min(rgb_img.height, crop_y1+req_h)
            dst_x, dst_y = max(0, -crop_x1), max(0, -crop_y1)
            
            if src_x2 > src_x1 and src_y2 > src_y1:
                region = rgb_img.crop((src_x1, src_y1, src_x2, src_y2))
                canvas.paste(region, (dst_x, dst_y))
            rgb_img = canvas

        # 5. Final Resize & Compress
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
        
    except Exception as e:
        return None, str(e)

# --- 7. UI FLOW ---
st.markdown("<h1>Global Passport Pro AI ✨</h1>", unsafe_allow_html=True)

if st.session_state.step == 1:
    st.markdown("### 📋 Step 1: Config & Upload")
    
    df = pd.DataFrame(PHOTO_STANDARDS).T.reset_index()
    df = df.rename(columns={"index": "Country", "mm": "Size", "w": "W", "h": "H", "kb": "Max KB"})
    st.dataframe(df[["Country", "Size", "W", "H", "Max KB"]], hide_index=True, use_container_width=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        selected = st.selectbox("Country:", list(PHOTO_STANDARDS.keys()))
        st.session_state.selected_std = selected
    with col_b:
        q_mode = st.selectbox("Quality:", ["Standard (~250 KB)", "Max Quality (Uncompressed)", "Strict Upload (< 100 KB)"])
        st.session_state.target_quality = q_mode
    
    tab_up, tab_cam = st.tabs(["📤 Upload", "📸 Camera"])
    
    # INPUT HANDLER
    img_buffer = None
    with tab_up:
        uploaded = st.file_uploader("Upload", type=['jpg','png','jpeg'], label_visibility="collapsed")
        if uploaded: img_buffer = uploaded
    with tab_cam:
        if not st.session_state.cam_active:
            if st.button("🔵 Open Camera"): st.session_state.cam_active = True; st.rerun()
        else:
            snap = st.camera_input("Selfie", label_visibility="collapsed")
            if snap: img_buffer = snap
            if st.button("❌ Close Camera"): st.session_state.cam_active = False; st.rerun()

    # AUTO-ANALYSIS ON UPLOAD
    if img_buffer:
        try:
            # Open and fix orientation
            img = Image.open(img_buffer)
            img = ImageOps.exif_transpose(img)
            st.session_state.input_image = img
            st.session_state.file_size_kb = img_buffer.size / 1024
            
            # Analyze
            with st.spinner("🔍 Analyzing image..."):
                res = analyze_image(img, st.session_state.file_size_kb, selected)
                st.session_state.validation_result = res

            # SHOW REPORT
            st.divider()
            col1, col2 = st.columns([1, 1.5])
            with col1:
                st.image(img, caption="Original", width=150)
            with col2:
                metrics = {
                    "Check": ["Dimensions", "Size", "Face Position"],
                    "Status": [
                        "✅ Pass" if res['dim_ok'] else "❌ Fail",
                        "✅ Pass" if res['size_ok'] else f"❌ {res['current_kb']:.0f} KB",
                        "✅ Pass" if res['icao_ok'] else "⚠️ Check"
                    ]
                }
                st.table(pd.DataFrame(metrics))

            # ACTION BUTTONS
            if res['is_compliant']:
                st.success("🎉 This photo matches all standards!")
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                st.download_button("⬇️ Download As-Is", buf.getvalue(), "compliant.jpg", "image/jpeg")
                st.write("--- OR ---")
            
            if st.button("✨ Auto-Fix & Generate Passport Photo"):
                st.session_state.step = 2; st.rerun()
                
        except Exception as e:
            st.error(f"Error reading image: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# STEP 2: PROCESSING
elif st.session_state.step == 2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ⚡ Step 2: AI Processing")
    
    with st.status("🚀 Processing...", expanded=True) as status:
        st.write("✂️ Removing background...")
        st.write("📏 Aligning face geometry...")
        
        buf, error_msg = process_photo(
            st.session_state.input_image, 
            PHOTO_STANDARDS[st.session_state.selected_std],
            st.session_state.target_quality
        )
        
        if buf is None:
            status.update(label="Failed!", state="error")
            st.error(f"Processing failed: {error_msg}")
            if st.button("⬅️ Go Back"): st.session_state.step = 1; st.rerun()
        else:
            # Success Path
            st.session_state.processed_image = buf
            st.session_state.final_size = error_msg # In success case, this is size
            status.update(label="Done!", state="complete", expanded=False)
            st.session_state.step = 3; st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)

# STEP 3: RESULT
elif st.session_state.step == 3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ✅ Step 3: Ready")
    
    st.image(st.session_state.processed_image, caption=f"Size: {st.session_state.final_size:.1f} KB", width=250)
    st.download_button("⬇️ Download Photo", st.session_state.processed_image, "passport.jpg", "image/jpeg")
    
    st.markdown(f'<br><a href="https://paypal.me/698789" target="_blank" class="paypal-btn">☕ Buy me a Coffee</a>', unsafe_allow_html=True)
    if st.button("🔄 Start Over"): st.session_state.step = 1; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)