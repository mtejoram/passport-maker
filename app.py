import streamlit as st
from rembg import remove
from PIL import Image
import io
import numpy as np
import cv2

# --- CONFIGURATION ---
TARGET_W, TARGET_H = 630, 810
MAX_FILE_SIZE_KB = 250

st.set_page_config(page_title="Indian Passport Photo Maker", page_icon="📸")

st.title("📸 Indian Passport Photo Maker")
st.write(
    """
    Upload your photo to automatically:
    1. Remove the background (White).
    2. Crop to Indian Passport standards (Head ~75% height).
    3. Resize to **630x810 pixels**.
    4. Compress to **<250KB**.
    """
)

def process_image(input_image):
    # 1. Remove Background
    with st.spinner("Removing background (AI)..."):
        # Convert PIL to bytes for rembg
        buf = io.BytesIO()
        input_image.save(buf, format="PNG")
        input_data = buf.getvalue()
        
        subject_data = remove(input_data)
        pil_image = Image.open(io.BytesIO(subject_data)).convert("RGBA")
        
        # Create White Background
        white_bg = Image.new("RGBA", pil_image.size, "WHITE")
        white_bg.paste(pil_image, (0, 0), pil_image)
        pil_rgb = white_bg.convert("RGB")

    # 2. Detect Face (OpenCV)
    opencv_image = np.array(pil_rgb)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
    
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return None, "No face detected. Please use a photo with better lighting."

    # Use largest face
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    
    # 3. Calculate Crop
    face_center_x = x + w // 2
    face_center_y = y + h // 2
    estimated_head_height = h * 1.5 # Padding for hair
    
    required_photo_height = int(estimated_head_height / 0.75)
    aspect_ratio = TARGET_W / TARGET_H
    required_photo_width = int(required_photo_height * aspect_ratio)
    
    crop_x1 = face_center_x - (required_photo_width // 2)
    crop_x2 = crop_x1 + required_photo_width
    
    vertical_shift = int(required_photo_height * 0.1)
    crop_y1 = (face_center_y - (required_photo_height // 2)) - vertical_shift
    crop_y2 = crop_y1 + required_photo_height
    
    # Safe Crop
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

    # 4. Resize
    final_image = final_crop.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
    
    # 5. Compress
    quality = 95
    while quality > 10:
        out_buffer = io.BytesIO()
        final_image.save(out_buffer, format="JPEG", quality=quality)
        size_kb = out_buffer.tell() / 1024
        if size_kb < MAX_FILE_SIZE_KB:
            out_buffer.seek(0)
            return out_buffer, None
        quality -= 5
    
    return None, "Could not compress image enough."

# --- UI ---
uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Photo", width=300)
    
    if st.button("Process Photo"):
        processed_img_buffer, error = process_image(image)
        
        if error:
            st.error(error)
        else:
            st.success("Photo processed successfully!")
            st.image(processed_img_buffer, caption="Passport Photo (630x810)", width=300)
            
            st.download_button(
                label="Download Passport Photo",
                data=processed_img_buffer,
                file_name="indian_passport_photo.jpg",
                mime="image/jpeg"
            )