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

# --- 2. CSS STYLING (Clean & Professional) ---
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
        .success-text { color: #00ff7f; font-weight: bold; }
        .fail-text { color: #ff4b4b; font-weight: bold; }
        
        div.stButton > button {
            width: 100%;
            border-radius: 8px;
            font-weight: bold;
            padding: 0.6rem 1rem;
            margin-top: 10px;
        }
        #MainMenu, footer, header {visibility: hidden;}
        .stFileUploader label, .stCameraInput label { display: none; }
    </style>
""", unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if 'step' not in st.session_state: st.session_state.step = 1
if 'input_image' not in st.session_state: st.session_state.input_image = None
if 'processed_image' not in st.session_state: st.session_state.processed_image = None
if 'cam_active' not in st.session_state: st.session_state.cam_active = False
if 'bg_mode' not in st.session_state: st.session_state.bg_mode = "Auto-Remove (White BG)"

# --- 4. UTILS ---
def resize_if_huge(img, max_dim=1500):
    w, h = img.size
    if w > max_dim or h > max_dim:
        ratio = min(max_dim/w, max_dim/h)
        return img.resize((int(w*ratio), int(h*ratio)), Image.Resampling.LANCZOS)
    return img

# --- 5. ICAO PROCESSING ENGINE ---
def process_photo(pil_img, std, bg_choice):
    try:
        # 0. Safety Resize
        work_img = resize_if_huge(pil_img)

        # 1. Background Handling
        if bg_choice == "Auto-Remove (White BG)":
            buf = io.BytesIO()
            work_img.save(buf, format="PNG")
            # Alpha matting for hair detail
            subject_mask = remove(buf.getvalue(), only_mask=True, alpha_matting=True, alpha_matting_foreground_threshold=240)
            mask_img = Image.open(io.BytesIO(subject_mask)).convert("L")
            
            foreground = work_img.convert("RGBA")
            bg = Image.new("RGBA", foreground.size, "WHITE")
            final_composite = Image.composite(foreground, bg, mask_img)
            rgb_img = final_composite.convert("RGB")
        else:
            # Keep Original: Use original pixels, but we still need detection
            rgb_img = work_img.convert("RGB")

        # 2. ICAO Face Detection & Cropping
        cv_img = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # Use High-Precision Haarcascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) > 0:
            # Get largest face
            fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            
            # --- STRICT ICAO LOGIC ---
            # 1. Center Line: Middle of the face
            center_x = fx + fw // 2
            
            # 2. Eye Line Estimate: Roughly 45% down from the top of the face box
            eye_line_y = fy + int(fh * 0.45)
            
            # 3. Chin Estimate: Bottom of face box
            chin_y = fy + fh
            
            # 4. Calculate Required Height based on Head Size
            # ICAO: Head (Crown to Chin) must be 70-80% of photo.
            # We estimate Crown is roughly 1.6x the eye-to-chin distance above chin.
            eye_to_chin = chin_y - eye_line_y
            head_height_est = eye_to_chin * 2.2 # Approximation of full head
            
            # Target: Head should be 75% (0.75) of image height
            req_img_h = int(head_height_est / 0.75)
            req_img_w = int(req_img_h * (std['w'] / std['h']))
            
            # 5. Determine Crop Box (Centering the Eye Line)
            # ICAO: Eyes should be roughly 60% from the bottom (or 40% from top)
            crop_y1 = eye_line_y - int(req_img_h * 0.4)
            crop_x1 = center_x - req_img_w // 2
            
        else:
            # Fallback: Center Crop if face fails
            req_img_w, req_img_h = rgb_img.width, rgb_img.height
            crop_x1, crop_y1 = 0, 0

        # Safe Crop (Paste onto white canvas if out of bounds)
        canvas = Image.new("RGB", (req_img_w, req_img_h), "WHITE")
        
        src_x1 = max(0, crop_x1)
        src_y1 = max(0, crop_y1)
        src_x2 = min(rgb_img.width, crop_x1 + req_img_w)
        src_y2 = min(rgb_img.height, crop_y1 + req_img_h)
        
        dst_x = max(0, -crop_x1)
        dst_y = max(0, -crop_y1)
        
        if src_x2 > src_x1 and src_y2 > src_y1:
            region = rgb_img.crop((src_x1, src_y1, src_x2, src_y2))
            canvas.paste(region, (dst_x, dst_y))
        
        final_img = canvas

        # 3. Final Resize to Exact Target
        final_output = final_img.resize((std['w'], std['h']), Image.Resampling.LANCZOS)
        
        # 4. Save (Max Quality)
        out = io.BytesIO()
        final_output.save(out, format="JPEG", quality=100, subsampling=0)
        out.seek(0)
        
        return out, out.tell() / 1024
        
    except Exception as e:
        return None, str(e)

# --- 6. SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Country Selection
    selected_country = st.selectbox("Target Country:", list(PHOTO_STANDARDS.keys()))
    st.session_state.selected_std = selected_country
    
    # Background Mode
    bg_mode = st.radio("Background Mode:", ["Auto-Remove (White BG)", "Keep Original (Hair Safe)"])
    st.session_state.bg_mode = bg_mode
    
    st.divider()
    st.info("ℹ️ 'Keep Original' is best if you already have a white wall behind you. It ensures hair details are perfect.")

# --- 7. MAIN INTERFACE ---
st.markdown("<h1 style='text-align: center;'>Global Passport Pro ✨</h1>", unsafe_allow_html=True)

if st.session_state.step == 1:
    
    # --- DIMENSION TABLE (Nice Expander) ---
    with st.expander("📊 View Size & Dimension Standards", expanded=False):
        df = pd.DataFrame(PHOTO_STANDARDS).T.reset_index()
        df = df.rename(columns={"index": "Country", "mm": "Size (mm)", "w": "Px Width", "h": "Px Height", "kb": "Max KB"})
        st.dataframe(df[["Country", "Size (mm)", "Px Width", "Px Height", "Max KB"]], hide_index=True, use_container_width=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # TABS
    tab_up, tab_cam = st.tabs(["📤 Upload Photo", "📸 Take Selfie"])
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

    # --- IMAGE PREVIEW & ACTION ---
    if img_buffer:
        img = Image.open(img_buffer)
        img = ImageOps.exif_transpose(img)
        st.session_state.input_image = img
        
        # Two column layout for "Before" state
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(img, caption="Original Upload", use_container_width=True)
            
        with col2:
            st.markdown("### 🔍 Action Required")
            st.markdown(f"**Target:** {selected_country}")
            st.markdown(f"**Mode:** {bg_mode}")
            
            # Primary Action Button aligned top
            if st.button("✨ Auto-Fix & Generate", type="primary"):
                st.session_state.step = 2; st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    
    # ICAO Guide below
    

# --- STEP 2: PROCESSING & RESULT ---
elif st.session_state.step == 2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # Run processing
    with st.spinner("🤖 AI is aligning face geometry to ICAO standards..."):
        # Artificial delay for UX
        time.sleep(0.8)
        
        buf, size_kb = process_photo(
            st.session_state.input_image,
            PHOTO_STANDARDS[st.session_state.selected_std],
            st.session_state.bg_mode
        )
        
        if buf:
            st.session_state.processed_image = buf
            st.session_state.final_size = size_kb
            st.session_state.step = 3; st.rerun()
        else:
            st.error(f"Processing failed: {size_kb}")
            if st.button("Back"): st.session_state.step = 1; st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 3: DOWNLOAD ---
elif st.session_state.step == 3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    col_res1, col_res2 = st.columns([1, 1])
    
    with col_res1:
        st.image(st.session_state.processed_image, caption="Final Passport Photo", use_container_width=True)
        
    with col_res2:
        st.markdown("### ✅ Ready!")
        st.markdown(f"**Specs:** {PHOTO_STANDARDS[st.session_state.selected_std]['w']}x{PHOTO_STANDARDS[st.session_state.selected_std]['h']} px")
        st.markdown(f"**Size:** {st.session_state.final_size:.1f} KB")
        
        st.download_button(
            label="⬇️ Download Image",
            data=st.session_state.processed_image,
            file_name="passport_photo.jpg",
            mime="image/jpeg",
            type="primary"
        )
        
        if st.button("🔄 Start Over"):
            st.session_state.step = 1; st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)