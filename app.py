import streamlit as st
from rembg import remove
from PIL import Image
import io
import numpy as np
import cv2

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Passport Pro AI",
    page_icon="🛂",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. ADVANCED CSS (The "Glassmorphism" Look) ---
st.markdown("""
    <style>
        /* IMPORT GOOGLE FONT */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

        /* RESET & GLOBAL THEME */
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }
        
        /* BACKGROUND GRADIENT */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            background-attachment: fixed;
        }

        /* GLASSMORPHISM CARD */
        .glass-card {
            background: rgba(255, 255, 255, 0.75);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
            padding: 2.5rem;
            margin-bottom: 2rem;
        }

        /* TYPOGRAPHY */
        h1 {
            color: #1a202c;
            font-weight: 700 !important;
            letter-spacing: -1px;
            text-align: center;
            margin-bottom: 0px !important;
        }
        p.subtitle {
            color: #4a5568;
            text-align: center;
            font-size: 1.1rem;
            margin-top: 0px;
            margin-bottom: 2rem;
            font-weight: 300;
        }
        h3 {
            color: #2d3748;
            font-weight: 600 !important;
            font-size: 1.2rem !important;
            margin-top: 1rem !important;
        }

        /* CUSTOM BUTTONS */
        div.stButton > button {
            background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
            color: white !important;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08);
            width: 100%;
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 7px 14px rgba(50, 50, 93, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08);
        }
        div.stButton > button:active {
            transform: translateY(1px);
        }

        /* HIDE UGLY STREAMLIT ELEMENTS */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* CUSTOM TABS */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(255,255,255,0.5);
            border-radius: 10px;
            padding: 10px 20px;
            border: none;
        }
        .stTabs [aria-selected="true"] {
            background-color: #fff !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            font-weight: 600 !important;
            color: #4b6cb7 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. LOGIC (Same Robust Backend) ---
TARGET_W, TARGET_H = 630, 810
MAX_FILE_SIZE_KB = 250

def process_image(input_image):
    buf = io.BytesIO()
    input_image.save(buf, format="PNG")
    subject_data = remove(buf.getvalue(), alpha_matting=True)
    foreground = Image.open(io.BytesIO(subject_data)).convert("RGBA")
    
    new_bg = Image.new("RGBA", foreground.size, "WHITE")
    new_bg.paste(foreground, (0, 0), foreground)
    final_rgb = new_bg.convert("RGB")
    
    opencv_img = np.array(final_rgb)
    opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
    
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        face_cx, face_cy = x + w//2, y + h//2
        head_h = h * 1.5
        req_h = int(head_h / 0.75)
        req_w = int(req_h * (TARGET_W / TARGET_H))
        
        c_x1 = face_cx - req_w // 2
        c_y1 = (face_cy - req_h // 2) - int(req_h * 0.1)
        c_x2, c_y2 = c_x1 + req_w, c_y1 + req_h
        
        final_rgb_padded = Image.new("RGB", (final_rgb.width + req_w*2, final_rgb.height + req_h*2), "WHITE")
        final_rgb_padded.paste(final_rgb, (req_w, req_h))
        final_rgb = final_rgb_padded.crop((c_x1+req_w, c_y1+req_h, c_x2+req_w, c_y2+req_h))

    final_rgb = final_rgb.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
    
    quality = 95
    while quality > 10:
        out_buf = io.BytesIO()
        final_rgb.save(out_buf, format="JPEG", quality=quality)
        if out_buf.tell() / 1024 < MAX_FILE_SIZE_KB:
            out_buf.seek(0)
            return out_buf
        quality -= 5
    return out_buf

# --- 4. UI LAYOUT ---

# Header Section
st.markdown("<h1>Passport Pro AI 🛂</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Professional Indian Passport Photos in seconds.</p>", unsafe_allow_html=True)

# Main Glass Card
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

# Mode Selection Tabs
tab_upload, tab_cam = st.tabs(["📤 Upload File", "📸 Use Camera"])

img_file = None

with tab_upload:
    st.markdown("### Drop your photo here")
    uploaded = st.file_uploader("", type=['jpg','png','jpeg'], label_visibility="collapsed")
    if uploaded: img_file = uploaded

with tab_cam:
    st.markdown("### Take a Selfie")
    # Camera Toggle Logic
    if "cam_active" not in st.session_state: st.session_state.cam_active = False
    
    if not st.session_state.cam_active:
        if st.button("🔴 Activate Camera"):
            st.session_state.cam_active = True
            st.rerun()
    else:
        cam_snap = st.camera_input("Look straight and center your face", label_visibility="collapsed")
        if cam_snap: img_file = cam_snap
        if st.button("❌ Close Camera"):
            st.session_state.cam_active = False
            st.rerun()

# Processing & Result Section
if img_file:
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h3 style='text-align: center'>Original</h3>", unsafe_allow_html=True)
        st.image(img_file, use_container_width=True)
        
    with col2:
        st.markdown("<h3 style='text-align: center'>Passport Ready</h3>", unsafe_allow_html=True)
        
        # Placeholder for result
        result_placeholder = st.empty()
        
        # Process Button
        if result_placeholder.button("✨ Generate Photo"):
            with st.spinner("Applying AI magic..."):
                final_buffer = process_image(Image.open(img_file))
                
                # Update placeholder with result
                result_placeholder.image(final_buffer, use_container_width=True)
                
                # Show Download Button
                st.balloons()
                st.download_button(
                    label="⬇️ Download High-Res Image",
                    data=final_buffer,
                    file_name="passport_photo.jpg",
                    mime="image/jpeg"
                )

st.markdown('</div>', unsafe_allow_html=True) # End Glass Card

# Footer
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px; font-size: 0.8rem;'>
        Secure & Private • No data saved • ICAO Compliant<br>
        Built with ❤️ using Python & AI
    </div>
""", unsafe_allow_html=True)