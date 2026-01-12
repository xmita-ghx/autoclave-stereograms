import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import io

def apply_custom_theme():
    st.markdown("""
        <style>
        /* Main background and text */
        .stApp {
            background-color: #000000;
            color: #FFFFFF;
        }
        
        /* Headers and text color */
        h1, h2, h3, p, span, label {
            color: #FFFFFF !important;
        }

        /* Gold borders for image containers and uploaders */
        div[data-testid="stFileUploader"], .stImage > img {
            border: 2px solid #D4AF37 !important;
            border-radius: 10px;
        }

        /* Light Purple Buttons */
        div.stButton > button {
            background-color: #A020F0 !important;
            color: white !important;
            border: 1px solid #D4AF37;
            border-radius: 5px;
            font-weight: bold;
        }
        
        div.stButton > button:hover {
            background-color: #DDA0DD !important; /* Lighter purple on hover */
            border: 1px solid #FFFFFF;
        }

        /* Radio button text color */
        div[data-testid="stRadio"] label {
            color: #FFFFFF !important;
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.to(device)
    model.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    return model, transforms, device

def enhance_contrast(img):
    img_np = np.array(img)
    # Convert to lab color space for better contrast manipulation
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced_rgb)

def make_sbs(img, depth, max_shift=15):
    w, h = img.size
    img_np = np.array(img)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    left_eye = np.zeros_like(img_np)
    right_eye = np.zeros_like(img_np)
    for y in range(h):
        row_shifts = (depth_norm[y] * max_shift).astype(int)
        for x in range(w):
            s = row_shifts[x]
            if x - s >= 0: left_eye[y, x-s] = img_np[y, x]
            if x + s < w: right_eye[y, x+s] = img_np[y, x]
    mask_l = np.all(left_eye == 0, axis=-1)
    left_eye[mask_l] = img_np[mask_l]
    mask_r = np.all(right_eye == 0, axis=-1)
    right_eye[mask_r] = img_np[mask_r]
    return Image.fromarray(np.hstack((left_eye, right_eye)))

def make_magic_eye(depth, pattern_div=8):
    h, w = depth.shape
    pattern_w = w // pattern_div
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    shifts = (depth_norm * (pattern_w // 4)).astype(int)
    output = np.random.randint(0, 255, (h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(pattern_w, w):
            output[y, x] = output[y, x - pattern_w + shifts[y, x]]
    return Image.fromarray(output)

st.set_page_config(page_title="3D Stereogram Creator", layout="centered")
apply_custom_theme()

st.title("AUTOCLAVE:")
st.title("2D TO 3D STEREOGRAM CONVERTER")

file = st.file_uploader("Upload Image (MAXIMUM: 5MB)   [Start with simple shapes! They should be high contrast images!]", type=['jpg', 'jpeg', 'png'])

if file:
    MAX_FILE_SIZE = 5 * 1024 * 1024 
    if file.size > MAX_FILE_SIZE:
        st.error(f"FILE TOO LARGE: Limit is 5MB.")
    else:
        img = Image.open(file).convert("RGB")
        

        st.image(img, caption="Original Image", width=300)
        
        mode = st.radio("Choose 3D Mode:", ["Side-by-Side (Stereopair)", "Autostereogram (Hidden)"])
        
        if st.button("Generate 3D View"):
            with st.spinner("Enhancing contrast & calculating depth..."):
                
                enhanced_img = enhance_contrast(img)
                
                
                model, transform, device = load_model()
                img_cv = cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_RGB2BGR)
                input_batch = transform(img_cv).to(device)
                
                with torch.no_grad():
                    prediction = model(input_batch)
                    depth_map = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1), size=img_cv.shape[:2], mode="bicubic"
                    ).squeeze().cpu().numpy()

                
                if mode == "Side-by-Side (Stereopair)":
                    result = make_sbs(img, depth_map)
                    st.subheader("Your 3D Result")
                    st.image(result, use_container_width=True)
                else:
                    st.subheader("AI Generated Depth Map")
                    depth_vis = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
                    st.image(depth_vis, caption="Outline for the Stereogram", width=300)
                    
                    result = make_magic_eye(depth_map)
                    st.subheader("Your Stereogram Result!")
                    st.image(result, use_container_width=True, channels="L")
                
                st.success("Done! You can right-click to save.")