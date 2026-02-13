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
st.set_page_config(page_title="Global Passport Pro", page_icon="🛂", layout="wide")

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
        .sidebar-text { font-size: 0.85rem; color: #e0e0e0; }
        .success-box { border-left: 5px solid #00ff7f; background: rgba(0, 255, 127, 0.1); padding: 15px; border-radius: 5px; }
        .fail-box { border-left: 5px solid #ff4b4b; background: rgba(255, 75, 75, 0.1); padding: 15px; border-radius: 5px; }
        
        div.stButton > button {
            width: 100%;
            border-radius: 8px;
            font-weight: bold;
            padding: 0.5rem 1rem;
        }
        .paypal-btn {
            background: #FFC439; color: black !important; padding: 12px 30px; 
            border-radius: 50px; text-decoration: none; font-weight: bold; display: inline-block;
        }
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
if 'bg_mode' not in st.session_state: st.session_state.bg_mode = "Auto-Remove (White BG)"

# --- 4. SAFETY UTILS ---
def resize_if_huge(img, max_dim=1500):
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
def process_photo(pil_img, std, quality_mode, bg_choice):
    try:
        # 0. Safety Resize
        work_img = resize_if_huge(pil_img)

        # 1. Determine Mask (Strictly for detection if keeping BG)
        if bg_choice == "Auto-Remove (White BG)":
            buf = io.BytesIO()
            work_img.save(buf, format="PNG")
            subject_mask = remove(buf.getvalue(), only_mask=True, alpha_matting=True)
            mask_img = Image.open(io.BytesIO(subject_mask)).convert("L")
            
            foreground = work_img.convert("RGBA")
            bg = Image.new("RGBA", foreground.size, "WHITE")
            final_composite = Image.composite(foreground, bg, mask_img)
            rgb_img = final_composite.convert("RGB")
            
            # Use mask for cropping
            mask_np = np.array(mask_img)
            
        else:
            # "Keep Original" - BYPASS REMBG ENTIRELY FOR PIXELS
            # We only use face detection for cropping coordinates
            rgb_img = work_img.convert("RGB")
            # Create a dummy full mask so we rely 100% on Face Detect
            mask_np = np.ones((rgb_img.height, rgb_img.width), dtype=np.uint8) * 255

        # 2. Geometry Calculation
        # Find face using OpenCV
        cv_img = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.1, 5)
        
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            cx = x + w // 2
            chin_y = y + h
            # Estimate top of head (hair) based on face box
            # Usually top of hair is ~0.5 to 0.8 face heights above the face box top
            top_y = max(0, y - int(h * 0.6))
        else:
            # Fallback center crop
            cx = rgb_img.width // 2
            top_y = 0
            chin_y = rgb_img.height

        head_h = chin_y - top_y
        if head_h < 1: head_h = rgb_img.height // 2
            
        req_h = int(head_h / 0.75) 
        req_w = int(req_h * (std['w'] / std['h']))
        
        # Calculate Crop Box
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

        # 3. Final Resize & Compress
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

# --- 7. UI SIDEBAR (STANDARDS TABLE) ---
with st.sidebar:
    st.markdown("### 📋 Requirements")
    df = pd.DataFrame(PHOTO_STANDARDS).T.reset_index()
    df = df.rename(columns={"index": "Country", "mm": "Size", "kb": "Max KB"})
    st.dataframe(df[["Country", "Size", "Max KB"]], hide_index=True, use_container_width=True)
    
    st.divider()
    st.markdown("### ⚙️ Settings")
    selected = st.selectbox("1. Country:", list(PHOTO_STANDARDS.keys()))
    st.session_state.selected_std = selected
    
    bg_choice = st.radio("2. Background:", ["Auto-Remove (White BG)", "Keep Original (Fixes Hair)"])
    st.session_state.bg_mode = bg_choice
    
    q_mode = st.selectbox("3. Quality:", ["Standard (~250 KB)", "Max Quality (Uncompressed)", "Strict Upload (< 100 KB)"])
    st.session_state.target_quality = q_mode

# --- 8. MAIN UI FLOW ---
st.markdown("<h1 style='text-align: center;'>Passport AI ✨</h1>", unsafe_allow_html=True)

if st.session_state.step == 1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    tab_up, tab_cam = st.tabs(["📤 Upload", "📸 Camera"])
    
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

    if img_buffer:
        try:
            img = Image.open(img_buffer)
            img = ImageOps.exif_transpose(img)
            st.session_state.input_image = img
            st.session_state.file_size_kb = img_buffer.size / 1024
            
            with st.spinner("🔍 Auditing Image..."):
                res = analyze_image(img, st.session_state.file_size_kb, selected)
                st.session_state.validation_result = res

            # --- TOP SECTION: ACTION BUTTONS ---
            col_actions, col_preview = st.columns([1, 1])
            
            with col_actions:
                st.subheader("Analysis Result")
                if res['is_compliant']:
                    st.success("✅ This photo is perfect!")
                    buf = io.BytesIO(); img.save(buf, format="JPEG")
                    st.download_button("⬇️ Download As-Is", buf.getvalue(), "compliant.jpg", "image/jpeg", use_container_width=True)
                else:
                    st.error("⚠️ Adjustments Needed")
                    btn_label = f"✨ Auto-Fix ({'Keep BG' if 'Keep' in bg_choice else 'White BG'})"
                    if st.button(btn_label, type="primary", use_container_width=True):
                        st.session_state.step = 2; st.rerun()

                # Metrics Table (Compact)
                metrics = {
                    "Check": ["Dimensions", "Size", "Face"],
                    "Status": [
                        "✅ Pass" if res['dim_ok'] else "❌ Fail",
                        "✅ Pass" if res['size_ok'] else f"❌ {res['current_kb']:.0f}KB",
                        "✅ Pass" if res['icao_ok'] else "⚠️ Check"
                    ]
                }
                st.table(pd.DataFrame(metrics))

            with col_preview:
                st.image(img, caption="Preview", use_container_width=True)
                
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.step == 2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    with st.status("🚀 Processing...", expanded=True) as status:
        if "Keep" in st.session_state.bg_mode:
            st.write("🛡️ Preserving original background (Hair Safe Mode)...")
        else:
            st.write("✂️ Removing background...")
            
        st.write("📏 Aligning geometry...")
        
        buf, error_msg = process_photo(
            st.session_state.input_image, 
            PHOTO_STANDARDS[st.session_state.selected_std],
            st.session_state.target_quality,
            st.session_state.bg_mode
        )
        
        if buf is None:
            status.update(label="Failed!", state="error")
            st.error(f"Error: {error_msg}")
            if st.button("⬅️ Go Back"): st.session_state.step = 1; st.rerun()
        else:
            st.session_state.processed_image = buf
            st.session_state.final_size = error_msg
            status.update(label="Done!", state="complete", expanded=False)
            st.session_state.step = 3; st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.step == 3:
    st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
    st.markdown("### ✅ Ready to Download")
    
    st.image(st.session_state.processed_image, width=300)
    st.caption(f"Final Size: {st.session_state.final_size:.1f} KB")
    
    st.download_button("⬇️ Download Photo", st.session_state.processed_image, "passport.jpg", "image/jpeg", type="primary")
    
    st.markdown(f'<br><a href="https://paypal.me/698789" target="_blank" class="paypal-btn">☕ Buy me a Coffee</a>', unsafe_allow_html=True)
    if st.button("🔄 Start Over"): st.session_state.step = 1; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)