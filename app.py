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
if 'bg_mode' not in st.session_state: st.session_state.bg_mode = "Auto-Remove (AI)"

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

        # 1. Determine Mask (We ALWAYS need the mask for crop coordinates, even if we keep BG)
        buf = io.BytesIO()
        work_img.save(buf, format="PNG")
        
        # Get mask only first
        subject_mask = remove(buf.getvalue(), only_mask=True, alpha_matting=True)
        mask_img = Image.open(io.BytesIO(subject_mask)).convert("L") # Grayscale mask
        
        # Prepare the image layer based on user choice
        if bg_choice == "Auto-Remove (White BG)":
            # Apply mask to create white background
            foreground = work_img.convert("RGBA")
            bg = Image.new("RGBA", foreground.size, "WHITE")
            # Use mask to paste foreground onto white
            final_composite = Image.composite(foreground, bg, mask_img)
            rgb_img = final_composite.convert("RGB")
        else:
            # "Keep Original" - Just use the original image
            rgb_img = work_img.convert("RGB")

        # 2. Smart Crop Calculation (Using the Mask)
        # We use the mask to find head position, even if we don't remove the BG
        mask_np = np.array(mask_img)
        rows = np.where(np.max(mask_np, axis=1) > 0)[0]
        
        if len(rows) > 0:
            top_y, bottom_y = rows[0], rows[-1]
            
            # Face Detect for X-centering
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

# --- 7. UI FLOW ---
st.markdown("<h1>Global Passport Pro AI ✨</h1>", unsafe_allow_html=True)

if st.session_state.step == 1:
    st.markdown("### 📋 Step 1: Config & Upload")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        selected = st.selectbox("1. Country:", list(PHOTO_STANDARDS.keys()))
        st.session_state.selected_std = selected
    with col_b:
        q_mode = st.selectbox("2. Quality:", ["Standard (~250 KB)", "Max Quality (Uncompressed)", "Strict Upload (< 100 KB)"])
        st.session_state.target_quality = q_mode
        
    # NEW BACKGROUND CONTROL
    bg_choice = st.radio("3. Background Processing:", ["Auto-Remove (White BG)", "Keep Original (Fixes Hair Issues)"], horizontal=True)
    st.session_state.bg_mode = bg_choice
    
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
            
            with st.spinner("🔍 Analyzing..."):
                res = analyze_image(img, st.session_state.file_size_kb, selected)
                st.session_state.validation_result = res

            st.divider()
            col1, col2 = st.columns([1, 1.5])
            with col1: st.image(img, caption="Original", width=150)
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

            if res['is_compliant']:
                st.success("🎉 Perfect Match!")
                buf = io.BytesIO(); img.save(buf, format="JPEG")
                st.download_button("⬇️ Download Original", buf.getvalue(), "compliant.jpg", "image/jpeg")
                st.write("--- OR ---")
            
            btn_label = "✨ Auto-Fix (Keep BG)" if bg_choice == "Keep Original (Fixes Hair Issues)" else "✨ Auto-Fix (White BG)"
            if st.button(btn_label):
                st.session_state.step = 2; st.rerun()
                
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.step == 2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ⚡ Step 2: Processing")
    
    with st.status("🚀 Processing...", expanded=True) as status:
        if st.session_state.bg_mode == "Auto-Remove (White BG)":
            st.write("✂️ Removing background...")
        else:
            st.write("🛡️ Preserving original background...")
            
        st.write("📏 Aligning geometry...")
        
        buf, error_msg = process_photo(
            st.session_state.input_image, 
            PHOTO_STANDARDS[st.session_state.selected_std],
            st.session_state.target_quality,
            st.session_state.bg_mode # Passing the user choice
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
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ✅ Step 3: Ready")
    st.image(st.session_state.processed_image, caption=f"Size: {st.session_state.final_size:.1f} KB", width=250)
    st.download_button("⬇️ Download Photo", st.session_state.processed_image, "passport.jpg", "image/jpeg")
    st.markdown(f'<br><a href="https://paypal.me/698789" target="_blank" class="paypal-btn">☕ Buy me a Coffee</a>', unsafe_allow_html=True)
    if st.button("🔄 Start Over"): st.session_state.step = 1; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)