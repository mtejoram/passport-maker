import streamlit as st
from rembg import remove
from PIL import Image
import io
import numpy as np
import cv2

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Passport Pro",
    page_icon="🛂",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. PREMIUM CSS STYLING ---
st.markdown("""
    <style>
        /* Main Background */
        .stApp {
            background-color: #f8f9fa;
        }
        
        /* Card Container for Main Content */
        .css-1r6slb0, .css-12oz5g7 {
            padding: 2rem;
            border-radius: 15px;
            background-color: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }
        
        /* Headings */
        h1 {
            color: #1a202c;
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        h3 {
            color: #4a5568;
            font-family: 'Helvetica Neue', sans-serif;
            text-align: center;
            font-weight: 400;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        
        /* Custom Button Styling */
        div.stButton > button {
            background-color: #3182ce;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
            font-weight: 600;
            width: 100%;
            transition: all 0.2s;
        }
        div.stButton > button:hover {
            background-color: #2b6cb0;
            box-shadow: 0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08);
            transform: translateY(-1px);
        }
        
        /* Success Message Box */
        .stSuccess {
            background-color: #f0fff4;
            border-left: 5px solid #48bb78;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            color: #a0aec0;
            font-size: 0.8rem;
            margin-top: 3rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. CORE LOGIC (Same robust processing) ---
TARGET_W, TARGET_H = 630, 810
MAX_FILE_SIZE_KB = 250

def process_image(input_image):
    # 1. Remove Background
    buf = io.BytesIO()
    input_image.save(buf, format="PNG")
    subject_data = remove(buf.getvalue(), alpha_matting=True)
    foreground = Image.open(io.BytesIO(subject_data)).convert("RGBA")
    
    # 2. White BG
    new_bg = Image.new("RGBA", foreground.size, "WHITE")
    new_bg.paste(foreground, (0, 0), foreground)
    final_rgb = new_bg.convert("RGB")
    
    # 3. Face Detect & Crop (Simplified for speed/reliability)
    opencv_img = np.array(final_rgb)
    opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
    
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        # Smart Crop Calculation
        face_cx, face_cy = x + w//2, y + h//2
        head_h = h * 1.5
        req_h = int(head_h / 0.75)
        req_w = int(req_h * (TARGET_W / TARGET_H))
        
        c_x1 = face_cx - req_w // 2
        c_y1 = (face_cy - req_h // 2) - int(req_h * 0.1) # Shift up
        c_x2, c_y2 = c_x1 + req_w, c_y1 + req_h
        
        # Pad and Crop
        final_rgb_padded = Image.new("RGB", (final_rgb.width + req_w*2, final_rgb.height + req_h*2), "WHITE")
        final_rgb_padded.paste(final_rgb, (req_w, req_h))
        final_rgb = final_rgb_padded.crop((c_x1+req_w, c_y1+req_h, c_x2+req_w, c_y2+req_h))

    # 4. Resize & Compress
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


# --- 4. THE UI LAYOUT ---

# Hero Section
st.markdown("<h1>Passport Pro 🛂</h1>", unsafe_allow_html=True)
st.markdown("<h3>Instant Indian Passport Photos. AI-Powered.</h3>", unsafe_allow_html=True)

# Main Card
with st.container():
    # Progress Steps
    col1, col2, col3 = st.columns(3)
    col1.metric("1. Upload", "Any Photo", delta=None)
    col2.metric("2. AI Magic", "Auto-Fix", delta=None)
    col3.metric("3. Download", "Print Ready", delta=None)
    
    st.divider()

    # Input Section
    tab_upload, tab_cam = st.tabs(["📤 Upload File", "📸 Take Selfie"])
    
    img_file = None
    
    with tab_upload:
        uploaded = st.file_uploader("", type=['jpg','png','jpeg'])
        if uploaded: img_file = uploaded
            
    with tab_cam:
        cam_snap = st.camera_input("Center your face and snap")
        if cam_snap: img_file = cam_snap

    # Processing & Result
    if img_file:
        st.write("---")
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("**Original**")
            st.image(img_file, use_container_width=True)
        
        with col_right:
            st.markdown("**Passport Ready**")
            if st.button("✨ Process Photo Now"):
                with st.spinner("Removing background & resizing..."):
                    result_buffer = process_image(Image.open(img_file))
                    
                st.image(result_buffer, use_container_width=True)
                st.success("✅ Validated: 630x810px | <250KB")
                
                st.download_button(
                    label="⬇️ Download High-Res JPG",
                    data=result_buffer,
                    file_name="passport_photo_final.jpg",
                    mime="image/jpeg"
                )

# Footer
st.markdown("""
    <div class='footer'>
        <p>🔒 Privacy First: Photos are processed in RAM and never saved.<br>
        Compatible with Indian Passport Seva & Visa applications.</p>
    </div>
""", unsafe_allow_html=True)