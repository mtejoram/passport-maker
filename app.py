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

# --- 2. PREMIUM CSS (FIXED FOR TEXT VISIBILITY) ---
st.markdown("""
    <style>
        /* Force light theme for the whole app */
        [data-testid="stAppViewContainer"] {
            background-color: #f0f2f6;
        }
        
        /* Main Card Styling */
        .main-card {
            background-color: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            color: #333333; /* Dark Grey Text */
        }

        /* Force ALL text to be dark (overrides system Dark Mode) */
        h1, h2, h3, h4, p, li, .stMarkdown, .stMetricLabel {
            color: #1a202c !important; 
        }
        
        /* Metric Values (The big numbers) */
        .stMetricValue {
            color: #3182ce !important;
        }

        /* Buttons */
        div.stButton > button {
            background-color: #3182ce;
            color: white !important;
            border-radius: 8px;
            border: none;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
        }
        div.stButton > button:hover {
            background-color: #2b6cb0;
        }
        
        /* Hide the default "Stop Camera" text which looks ugly */
        button[kind="header"] {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. PROCESSING LOGIC ---
TARGET_W, TARGET_H = 630, 810
MAX_FILE_SIZE_KB = 250

def process_image(input_image):
    # 1. Remove BG
    buf = io.BytesIO()
    input_image.save(buf, format="PNG")
    subject_data = remove(buf.getvalue(), alpha_matting=True)
    foreground = Image.open(io.BytesIO(subject_data)).convert("RGBA")
    
    # 2. White BG
    new_bg = Image.new("RGBA", foreground.size, "WHITE")
    new_bg.paste(foreground, (0, 0), foreground)
    final_rgb = new_bg.convert("RGB")
    
    # 3. Detect & Crop
    opencv_img = np.array(final_rgb)
    opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
    
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        # Smart Crop
        face_cx, face_cy = x + w//2, y + h//2
        head_h = h * 1.5
        req_h = int(head_h / 0.75)
        req_w = int(req_h * (TARGET_W / TARGET_H))
        
        c_x1 = face_cx - req_w // 2
        c_y1 = (face_cy - req_h // 2) - int(req_h * 0.1)
        c_x2, c_y2 = c_x1 + req_w, c_y1 + req_h
        
        # Pad canvas if needed
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

# --- 4. UI LAYOUT ---

st.markdown("<h1 style='text-align: center; color: #1a202c;'>Passport Pro 🛂</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #4a5568;'>Instant Indian Passport Photos</h3>", unsafe_allow_html=True)

# Main Container
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True) # Start Card

    # Tabs
    tab_upload, tab_cam = st.tabs(["📤 Upload File", "📸 Take Selfie"])
    
    img_file = None
    
    # --- TAB 1: UPLOAD ---
    with tab_upload:
        st.write("### Option 1: Upload from Gallery")
        uploaded = st.file_uploader("", type=['jpg','png','jpeg'], key="uploader")
        if uploaded: 
            img_file = uploaded

    # --- TAB 2: CAMERA (FIXED: NO AUTO-POPUP) ---
    with tab_cam:
        st.write("### Option 2: Use Camera")
        
        # We use session state to track if camera should be on
        if "camera_active" not in st.session_state:
            st.session_state.camera_active = False

        if not st.session_state.camera_active:
            # Show a button INSTEAD of the camera first
            if st.button("🔴 Open Camera"):
                st.session_state.camera_active = True
                st.rerun()
        else:
            # Only show camera widget if button was clicked
            cam_snap = st.camera_input("Center your face and snap", key="camera")
            if cam_snap: 
                img_file = cam_snap
            
            # Option to close camera
            if st.button("❌ Close Camera"):
                st.session_state.camera_active = False
                st.rerun()

    # --- RESULT SECTION ---
    if img_file:
        st.markdown("---")
        st.write("### Preview & Process")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.image(img_file, caption="Original Photo", use_container_width=True)
        
        with col_right:
            if st.button("✨ Convert to Passport Photo"):
                with st.spinner("Processing... Please wait."):
                    result_buffer = process_image(Image.open(img_file))
                
                st.success("Done!")
                st.image(result_buffer, caption="Passport Ready", use_container_width=True)
                
                st.download_button(
                    label="⬇️ Download Image",
                    data=result_buffer,
                    file_name="passport_photo.jpg",
                    mime="image/jpeg"
                )
    
    st.markdown('</div>', unsafe_allow_html=True) # End Card

# Footer
st.markdown("<p style='text-align: center; margin-top: 20px; color: #666;'>Privacy Note: Photos are not saved.</p>", unsafe_allow_html=True)