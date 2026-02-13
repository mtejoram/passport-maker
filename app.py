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
        div.stButton > button {
            width: 100%;
            border-radius: 8px;
            font-weight: bold;
            padding: 0.6rem 1rem;
            margin-top: 10px;
        }
        /* Hide elements */
        #MainMenu, footer, header {visibility: hidden;}
        .stFileUploader label, .stCameraInput label { display: none; }
    </style>
""", unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if 'step' not in st.session_state: st.session_state.step = 1
if 'input_image' not in st.session_state: st.session_state.input_image = None
if 'processed_image' not in st.session_state: st.session_state.processed_image = None
if 'cam_active' not in st.session_state: st.session_state.cam_active = False
if 'custom_specs' not in st.session_state: st.session_state.custom_specs = {"w": 600, "h": 600, "kb": 250}

# --- 4. UTILS ---
def resize_if_huge(img, max_dim=1500):
    w, h = img.size
    if w > max_dim or h > max_dim:
        ratio = min(max_dim/w, max_dim/h)
        return img.resize((int(w*ratio), int(h*ratio)), Image.Resampling.LANCZOS)
    return img

# --- 5. ICAO PROCESSING ENGINE (HAIR SAFE) ---
def process_photo(pil_img, target_w, target_h, max_kb, bg_choice):
    try:
        # 0. Safety Resize
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
            # Largest face
            fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            
            center_x = fx + fw // 2
            
            # --- HAIR-SAFE LOGIC ---
            # Instead of placing eyes high (40%), we place them lower (48%)
            # This pushes the face down, leaving more room at the top for hair.
            eye_line_y = fy + int(fh * 0.45) # Estimate eye level within face box
            
            # Estimate full head height (Chin to Crown)
            # Chin is roughly bottom of box. Crown is roughly 1.5x face height above chin.
            chin_y = fy + fh
            head_height_est = (chin_y - eye_line_y) * 2.4 # Generous multiplier for hair
            
            # Calculate required canvas size to fit this head
            # Target: Head should be ~75% of image height
            req_img_h = int(head_height_est / 0.75)
            req_img_w = int(req_img_h * (target_w / target_h))
            
            # Anchor Crop: Place Eye Line at 48% from top (Safe Zone)
            crop_y1 = eye_line_y - int(req_img_h * 0.48)
            crop_x1 = center_x - req_img_w // 2
            
        else:
            # Fallback: Center Crop
            req_img_w, req_img_h = rgb_img.width, rgb_img.height
            crop_x1, crop_y1 = 0, 0

        # Safe Paste (Handle out of bounds)
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

        # 3. Final Resize
        final_output = final_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        # 4. Smart Compress Loop
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

# --- 6. SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # 1. Country Selector (Empty Default)
    options = ["Select a Country..."] + list(PHOTO_STANDARDS.keys()) + ["🛠️ Custom / Manual Input"]
    selected_option = st.selectbox("1. Target Standard:", options)
    
    target_specs = None
    
    # Logic for Custom vs Standard
    if selected_option == "Select a Country...":
        st.warning("👈 Please select a country to start.")
    
    elif selected_option == "🛠️ Custom / Manual Input":
        st.markdown("---")
        st.markdown("**Manual Specifications:**")
        c_w = st.number_input("Width (px)", value=600)
        c_h = st.number_input("Height (px)", value=600)
        c_kb = st.number_input("Max Size (KB)", value=250)
        target_specs = {"w": c_w, "h": c_h, "kb": c_kb, "desc": "Custom"}
        
    else:
        # Standard Selection
        target_specs = PHOTO_STANDARDS[selected_option]
        # Show read-only info
        st.info(f"📏 {target_specs['w']}x{target_specs['h']} px\n💾 Max {target_specs['kb']} KB")

    st.markdown("---")
    bg_mode = st.radio("2. Background Mode:", ["Auto-Remove (White BG)", "Keep Original (Hair Safe)"])
    
    # Save to session
    if target_specs:
        st.session_state.custom_specs = target_specs
        st.session_state.bg_mode = bg_mode

# --- 7. MAIN INTERFACE ---
st.markdown("<h1 style='text-align: center;'>Passport AI ✨</h1>", unsafe_allow_html=True)

# Step 1: Upload (Only if Country Selected)
if st.session_state.step == 1:
    
    if selected_option == "Select a Country...":
        st.markdown('<div class="glass-card" style="text-align: center; color: #aaa;">', unsafe_allow_html=True)
        st.markdown("### ⬅️ Start by selecting a country in the sidebar")
        st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
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

        # VALIDATION & PREVIEW LOGIC
        if img_buffer:
            img = Image.open(img_buffer)
            img = ImageOps.exif_transpose(img)
            st.session_state.input_image = img
            
            # Calc Metrics
            curr_w, curr_h = img.size
            curr_kb = img_buffer.size / 1024
            req = st.session_state.custom_specs
            
            is_perfect = (curr_w == req['w'] and curr_h == req['h'] and curr_kb <= req['kb'])
            
            # --- TWO COLUMN LAYOUT ---
            col_img, col_info = st.columns([1, 1.2])
            
            with col_img:
                st.image(img, caption="Original Upload", use_container_width=True)
            
            with col_info:
                if is_perfect:
                    st.success("✅ Perfect Match! No changes needed.")
                    st.markdown("Your photo already meets the requirements.")
                else:
                    st.error("⚠️ Action Required")
                    st.markdown(f"**Issues Detected:**")
                    
                    # Dynamic Error List
                    if curr_w != req['w'] or curr_h != req['h']:
                        st.markdown(f"- ❌ **Dimensions:** {curr_w}x{curr_h} (Need {req['w']}x{req['h']})")
                    if curr_kb > req['kb']:
                        st.markdown(f"- ❌ **File Size:** {curr_kb:.0f} KB (Limit {req['kb']} KB)")
                    
                    st.markdown(f"- ℹ️ **Mode:** {bg_mode}")
                
                # ACTION BUTTONS (Aligned Top)
                if st.button("✨ Auto-Fix & Generate", type="primary"):
                    st.session_state.step = 2; st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

# Step 2: Processing
elif st.session_state.step == 2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    with st.spinner("🤖 AI is aligning face and fixing dimensions..."):
        time.sleep(0.5)
        req = st.session_state.custom_specs
        
        buf, size_val = process_photo(
            st.session_state.input_image,
            req['w'], req['h'], req['kb'],
            st.session_state.bg_mode
        )
        
        if buf:
            st.session_state.processed_image = buf
            st.session_state.final_size = size_val
            st.session_state.step = 3; st.rerun()
        else:
            st.error(f"Error: {size_val}")
            if st.button("Back"): st.session_state.step = 1; st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)

# Step 3: Download
elif st.session_state.step == 3:
    st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1,1])
    with col1:
        st.image(st.session_state.processed_image, caption="Result", use_container_width=True)
    with col2:
        st.success("✅ Processing Complete")
        st.markdown(f"**Final Size:** {st.session_state.final_size:.1f} KB")
        
        st.download_button(
            label="⬇️ Download Photo",
            data=st.session_state.processed_image,
            file_name="passport_photo.jpg",
            mime="image/jpeg",
            type="primary"
        )
        
        if st.button("🔄 Process Another"):
            st.session_state.step = 1; st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)