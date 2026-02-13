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
# layout="centered" is actually better for mobile than "wide"
st.set_page_config(page_title="Global Passport Pro", page_icon="🛂", layout="centered")

# --- 2. CSS STYLING (Mobile Optimized) ---
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
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 20px;
        }
        /* Mobile-friendly Button Sizing */
        div.stButton > button {
            width: 100%;
            border-radius: 12px;
            font-weight: bold;
            padding: 0.75rem 1rem;
            margin-top: 10px;
            min-height: 50px; /* Touch friendly */
        }
        /* Status Text Colors */
        .success-text { color: #00ff7f; font-weight: bold; }
        .fail-text { color: #ff4b4b; font-weight: bold; }
        
        #MainMenu, footer, header {visibility: hidden;}
        .stFileUploader label, .stCameraInput label { display: none; }
    </style>
""", unsafe_allow_html=True)

# --- 3. SESSION STATE INITIALIZATION ---
if 'step' not in st.session_state: st.session_state.step = 1
if 'input_image' not in st.session_state: st.session_state.input_image = None
if 'processed_image' not in st.session_state: st.session_state.processed_image = None
if 'cam_active' not in st.session_state: st.session_state.cam_active = False
if 'bg_mode' not in st.session_state: st.session_state.bg_mode = "Auto-Remove (White BG)"
if 'custom_specs' not in st.session_state: st.session_state.custom_specs = {"w": 600, "h": 600, "kb": 250, "mm": "Custom"}
if 'selected_std' not in st.session_state: st.session_state.selected_std = "Select a Country..."
if 'target_quality' not in st.session_state: st.session_state.target_quality = "Standard (~250 KB)"
if 'final_size' not in st.session_state: st.session_state.final_size = 0

# --- 4. UTILS ---
def resize_if_huge(img, max_dim=1500):
    w, h = img.size
    if w > max_dim or h > max_dim:
        ratio = min(max_dim/w, max_dim/h)
        return img.resize((int(w*ratio), int(h*ratio)), Image.Resampling.LANCZOS)
    return img

# --- 5. ICAO PROCESSING ENGINE ---
def process_photo(pil_img, target_w, target_h, max_kb, bg_choice):
    try:
        work_img = resize_if_huge(pil_img)

        # 1. Background Handling
        if bg_choice == "Auto-Remove (White BG)":
            buf = io.BytesIO()
            work_img.save(buf, format="PNG")
            subject_mask = remove(buf.getvalue(), only_mask=True, alpha_matting=True, alpha_matting_foreground_threshold=240)
            mask_img = Image.open(io.BytesIO(subject_mask)).convert("L")
            
            foreground = work_img.convert("RGBA")
            bg = Image.new("RGBA", foreground.size, "WHITE")
            final_composite = Image.composite(foreground, bg, mask_img)
            rgb_img = final_composite.convert("RGB")
        else:
            rgb_img = work_img.convert("RGB")

        # 2. Face Detection
        cv_img = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            center_x = fx + fw // 2
            
            # ICAO Logic
            eye_line_y = fy + int(fh * 0.45) 
            chin_y = fy + fh
            head_height_est = (chin_y - eye_line_y) * 2.4 
            
            req_img_h = int(head_height_est / 0.75)
            req_img_w = int(req_img_h * (target_w / target_h))
            
            crop_y1 = eye_line_y - int(req_img_h * 0.48)
            crop_x1 = center_x - req_img_w // 2
        else:
            req_img_w, req_img_h = rgb_img.width, rgb_img.height
            crop_x1, crop_y1 = 0, 0

        # Safe Paste
        canvas = Image.new("RGB", (req_img_w, req_img_h), "WHITE")
        src_x1, src_y1 = max(0, crop_x1), max(0, crop_y1)
        src_x2 = min(rgb_img.width, crop_x1 + req_img_w)
        src_y2 = min(rgb_img.height, crop_y1 + req_img_h)
        dst_x, dst_y = max(0, -crop_x1), max(0, -crop_y1)
        
        if src_x2 > src_x1 and src_y2 > src_y1:
            region = rgb_img.crop((src_x1, src_y1, src_x2, src_y2))
            canvas.paste(region, (dst_x, dst_y))
        
        final_img = canvas

        # 3. Final Resize & Compress
        final_output = final_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        quality = 100
        while quality > 10:
            out = io.BytesIO()
            final_output.save(out, format="JPEG", quality=quality, subsampling=0)
            if out.tell() / 1024 < max_kb:
                out.seek(0)
                return out, out.tell() / 1024
            quality -= 5
            
        out.seek(0)
        return out, out.tell() / 1024
        
    except Exception as e:
        return None, str(e)

# --- 6. MAIN UI ---
st.markdown("<h1 style='text-align: center;'>Global Passport Pro ✨</h1>", unsafe_allow_html=True)

if st.session_state.step == 1:
    
    # --- CONFIG CARD (MOVED TO MAIN SCREEN) ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 1️⃣ Settings")
    
    options = ["Select a Country..."] + list(PHOTO_STANDARDS.keys()) + ["🛠️ Custom / Manual Input"]
    
    # Handle Session State Index Logic safely
    try:
        sel_index = options.index(st.session_state.selected_std)
    except ValueError:
        sel_index = 0
        
    selected_option = st.selectbox("Target Standard:", options, index=sel_index)
    st.session_state.selected_std = selected_option
    
    # Logic for Custom vs Standard
    target_specs = None
    if selected_option == "Select a Country...":
        st.info("👆 Select a country to enable upload.")
        
    elif selected_option == "🛠️ Custom / Manual Input":
        c1, c2, c3 = st.columns(3)
        with c1: c_w = st.number_input("W (px)", value=600)
        with c2: c_h = st.number_input("H (px)", value=600)
        with c3: c_kb = st.number_input("KB Limit", value=250)
        c_mm = st.text_input("Size label (e.g. 35x45mm)", "Custom Size")
        target_specs = {"w": c_w, "h": c_h, "kb": c_kb, "mm": c_mm}
        
    else:
        target_specs = PHOTO_STANDARDS[selected_option]
        st.success(f"**{target_specs['mm']}** | {target_specs['w']}x{target_specs['h']} px | Max {target_specs['kb']} KB")

    if target_specs:
        st.session_state.custom_specs = target_specs
        
        # Additional Options (Only show if country selected)
        c_bg, c_qual = st.columns(2)
        with c_bg:
            bg_mode = st.selectbox("Background:", ["Auto-Remove (White BG)", "Keep Original (Hair Safe)"], index=0 if st.session_state.bg_mode == "Auto-Remove (White BG)" else 1)
            st.session_state.bg_mode = bg_mode
        with c_qual:
            q_mode = st.selectbox("Quality:", ["Standard (~250 KB)", "Max Quality (Uncompressed)", "Strict Upload (< 100 KB)"])
            st.session_state.target_quality = q_mode
    
    st.markdown("</div>", unsafe_allow_html=True)

    # --- UPLOAD CARD (Conditional) ---
    if target_specs:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 2️⃣ Capture or Upload")
        
        tab_up, tab_cam = st.tabs(["📤 Upload File", "📸 Camera"])
        
        img_buffer = None
        with tab_up:
            uploaded = st.file_uploader("Upload", type=['jpg','png','jpeg'], label_visibility="collapsed")
            if uploaded: img_buffer = uploaded
        with tab_cam:
            if not st.session_state.cam_active:
                if st.button("🔵 Start Camera"): 
                    st.session_state.cam_active = True
                    st.rerun()
            else:
                snap = st.camera_input("Selfie", label_visibility="collapsed")
                if snap: img_buffer = snap
                if st.button("❌ Close Camera"): 
                    st.session_state.cam_active = False
                    st.rerun()

        # PREVIEW & VALIDATION
        if img_buffer:
            img = Image.open(img_buffer)
            img = ImageOps.exif_transpose(img)
            st.session_state.input_image = img
            
            curr_w, curr_h = img.size
            curr_kb = img_buffer.size / 1024
            req = st.session_state.custom_specs
            
            is_perfect = (curr_w == req['w'] and curr_h == req['h'] and curr_kb <= req['kb'])
            
            st.markdown("---")
            col_img, col_info = st.columns([1, 1.5])
            
            with col_img:
                st.image(img, caption="Original", use_container_width=True)
            
            with col_info:
                if is_perfect:
                    st.success("✅ Perfect! No changes needed.")
                else:
                    st.warning("⚠️ Adjustments Needed")
                    st.markdown(f"**Target:** {req['mm']} ({req['w']}x{req['h']}px)")
                    
                    if curr_w != req['w'] or curr_h != req['h']:
                        st.markdown(f"❌ **Dims:** {curr_w}x{curr_h} px")
                    if curr_kb > req['kb']:
                        st.markdown(f"❌ **Size:** {curr_kb:.0f} KB (Limit {req['kb']})")
                
                # FIXED: Action button visible immediately without scrolling
                if st.button("✨ Auto-Fix & Generate", type="primary"):
                    st.session_state.step = 2; st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 2: PROCESSING ---
elif st.session_state.step == 2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    with st.spinner("🤖 AI is aligning face and cropping..."):
        time.sleep(0.5)
        req = st.session_state.custom_specs
        buf, size_val = process_photo(st.session_state.input_image, req['w'], req['h'], req['kb'], st.session_state.bg_mode)
        
        if buf:
            st.session_state.processed_image = buf
            st.session_state.final_size = size_val
            st.session_state.step = 3; st.rerun()
        else:
            st.error(f"Error: {size_val}")
            if st.button("Back"): st.session_state.step = 1; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 3: RESULT ---
elif st.session_state.step == 3:
    st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
    st.markdown("### ✅ Ready!")
    
    st.image(st.session_state.processed_image, caption="Passport Photo", width=250)
    
    req = st.session_state.custom_specs
    st.success(f"Standard: {req['mm']} | {req['w']}x{req['h']} px | {st.session_state.final_size:.1f} KB")
    
    st.download_button(
        label="⬇️ Download Photo",
        data=st.session_state.processed_image,
        file_name="passport_photo.jpg",
        mime="image/jpeg",
        type="primary"
    )
    
    if st.button("🔄 Process Another"): st.session_state.step = 1; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
