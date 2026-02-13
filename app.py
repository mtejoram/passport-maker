import streamlit as st
from rembg import remove
from PIL import Image, ImageEnhance
import io
import numpy as np
import cv2

# --- PAGE CONFIGURATION (Browser Tab Title & Icon) ---
st.set_page_config(
    page_title="Passport Pro AI",
    page_icon="🛂",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1 {
            text-align: center;
            color: #2E86C1;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            background-color: #2E86C1;
            color: white;
        }
        .css-1v0mbdj {
            display: flex;
            justify-content: center;
        }
        img {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# --- CONFIGURATION CONSTANTS ---
TARGET_W, TARGET_H = 630, 810
MAX_FILE_SIZE_KB = 250

def process_image(input_image):
    """
    Core logic: Remove BG -> Detect Face -> Crop -> Resize -> Compress
    """
    # 1. REMOVE BACKGROUND
    with st.spinner("✨ Removing background & Enhancing..."):
        # Convert to bytes
        buf = io.BytesIO()
        input_image.save(buf, format="PNG")
        input_data = buf.getvalue()
        
        # AI Magic
        subject_data = remove(input_data, alpha_matting=True) # High quality cut
        pil_image = Image.open(io.BytesIO(subject_data)).convert("RGBA")
        
        # Create Professional White Background
        white_bg = Image.new("RGBA", pil_image.size, "WHITE")
        white_bg.paste(pil_image, (0, 0), pil_image)
        pil_rgb = white_bg.convert("RGB")

    # 2. FACE DETECTION
    opencv_image = np.array(pil_rgb)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
    
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return None, "⚠️ No face detected. Please ensure good lighting and face the camera directly."

    # Use largest face
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    
    # 3. SMART CROP (ICAO Standards)
    face_center_x = x + w // 2
    face_center_y = y + h // 2
    estimated_head_height = h * 1.55  # Slightly more headroom for professional look
    
    required_photo_height = int(estimated_head_height / 0.75)
    aspect_ratio = TARGET_W / TARGET_H
    required_photo_width = int(required_photo_height * aspect_ratio)
    
    crop_x1 = face_center_x - (required_photo_width // 2)
    crop_x2 = crop_x1 + required_photo_width
    
    # Shift slightly up (Eyes at ~60% height)
    vertical_shift = int(required_photo_height * 0.1)
    crop_y1 = (face_center_y - (required_photo_height // 2)) - vertical_shift
    crop_y2 = crop_y1 + required_photo_height
    
    # Safe Crop Logic
    final_crop = Image.new("RGB", (required_photo_width, required_photo_height), (255, 255, 255))
    orig_w, orig_h = pil_rgb.size
    
    paste_x = max(0, -crop_x1)
    paste_y = max(0, -crop_y1)
    cut_x1 = max(0, crop_x1)
    cut_y1 = max(0, crop_y1)
    cut_x2 = min(orig_w, crop_x2)
    cut_y2 = min(orig_h, crop_y2)
    
    if cut_x2 > cut_x1 and cut_y2 > cut_y1:
        region = pil_rgb.crop((cut_x1, cut_y1, cut_x2, cut_y2))
        final_crop.paste(region, (paste_x, paste_y))

    # 4. RESIZE
    final_image = final_crop.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
    
    # 5. COMPRESS LOOP
    quality = 95
    while quality > 10:
        out_buffer = io.BytesIO()
        final_image.save(out_buffer, format="JPEG", quality=quality)
        size_kb = out_buffer.tell() / 1024
        if size_kb < MAX_FILE_SIZE_KB:
            out_buffer.seek(0)
            return out_buffer, None
        quality -= 5
    
    return None, "Image too complex to compress under 250KB."

# --- MAIN UI ---
st.title("🛂 Passport Pro AI")
st.markdown("### The easiest way to get an Indian Passport photo.")

# Create Tabs for cleaner interface
tab1, tab2, tab3 = st.tabs(["📤 Upload Photo", "📸 Take Selfie", "ℹ️ Guide"])

image_to_process = None

# TAB 1: UPLOAD
with tab1:
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], help="Upload a clear photo")
    if uploaded_file:
        image_to_process = Image.open(uploaded_file)

# TAB 2: CAMERA (Mobile Friendly!)
with tab2:
    camera_photo = st.camera_input("Take a picture")
    if camera_photo:
        image_to_process = Image.open(camera_photo)

# TAB 3: INSTRUCTIONS
with tab3:
    st.info("💡 **Tips for the perfect shot:**")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("✅ **Do:**")
        st.write("- Look straight at the camera.")
        st.write("- Keep a neutral expression.")
        st.write("- Use even lighting (no shadows).")
    with col_b:
        st.write("❌ **Don't:**")
        st.write("- Wear glasses (glare is bad).")
        st.write("- Wear white clothes (blends with bg).")
        st.write("- Tilt your head.")

st.markdown("---")

# --- PROCESSING SECTION ---
if image_to_process:
    # Show Original
    st.markdown("#### Preview")
    st.image(image_to_process, caption="Original", width=250)
    
    # Process Button
    if st.button("✨ Generate Passport Photo"):
        processed_buffer, error = process_image(image_to_process)
        
        if error:
            st.error(error)
        else:
            st.success("✅ Photo Ready! (630x810 px | White Bg | <250KB)")
            
            # Show Result
            st.image(processed_buffer, caption="Final Passport Photo", width=250)
            
            # Download Button
            st.download_button(
                label="⬇️ Download Image",
                data=processed_buffer,
                file_name="indian_passport_photo.jpg",
                mime="image/jpeg",
                use_container_width=True
            )