import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import io

# 1. SETUP & AI MODEL LOADING
@st.cache_resource
def load_model():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # MiDaS small is used for speed, but we will post-process for sharpness
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.to(device)
    model.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    return model, transforms, device

# 2. CONVERSION LOGIC: SIDE-BY-SIDE (SBS) STEREOPAIR
def make_sbs(img, depth, max_shift=15):
    w, h = img.size
    img_np = np.array(img)
    
    # Sharp Normalization: Ensures the outline is tight
    depth_min = depth.min()
    depth_max = depth.max()
    depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)
    
    left_eye = np.zeros_like(img_np)
    right_eye = np.zeros_like(img_np)

    for y in range(h):
        # Shifts are calculated per pixel to follow the outline exactly
        row_shifts = (depth_norm[y] * max_shift).astype(int)
        for x in range(w):
            s = row_shifts[x]
            if x - s >= 0: left_eye[y, x-s] = img_np[y, x]
            if x + s < w: right_eye[y, x+s] = img_np[y, x]
            
    # Fill gaps (artifacting) with original pixels to keep outline clean
    mask_l = np.all(left_eye == 0, axis=-1)
    left_eye[mask_l] = img_np[mask_l]
    mask_r = np.all(right_eye == 0, axis=-1)
    right_eye[mask_r] = img_np[mask_r]
            
    return Image.fromarray(np.hstack((left_eye, right_eye)))

# 3. CONVERSION LOGIC: AUTOSTEREOGRAM (HIDDEN)
def make_magic_eye(depth, pattern_div=8):
    h, w = depth.shape
    pattern_w = w // pattern_div
    
    # Normalize
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    shifts = (depth_norm * (pattern_w // 4)).astype(int)
    
    # Create high-detail noise pattern
    output = np.random.randint(0, 255, (h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(pattern_w, w):
            output[y, x] = output[y, x - pattern_w + shifts[y, x]]
    return Image.fromarray(output)

# 4. STREAMLIT UI
st.set_page_config(page_title="3D Stereogram Creator")
st.title("AUTOCLAVE:")
st.title("2D TO 3D STEREOGRAM CONVERTER")

# Updated label to explicitly show the 5MB limit
file = st.file_uploader("Upload Image (MAXIMUM: 5MB) [Start with simple shapes! They should be high contrast images!]", type=['jpg', 'jpeg', 'png'])

if file:
    # --- STRICT SIZE RANGE CHECK ---
    # 5MB = 5 * 1024 * 1024 bytes
    MAX_FILE_SIZE = 5 * 1024 * 1024 
    
    if file.size > MAX_FILE_SIZE:
        st.error(f"FILE TOO LARGE: Your file is {file.size / (1024*1024):.2f}MB. The strict limit is 5MB.")
    else:
        # Proceed only if file above 5MB
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Original Image", width=300)
        
        mode = st.radio("Choose 3D Mode:", ["Side-by-Side (Stereopair)", "Autostereogram (Hidden)"])
        
        if st.button("Generate 3D View"):
            with st.spinner("AI is calculating depth map..."):
                model, transform, device = load_model()
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                input_batch = transform(img_cv).to(device)
                
                with torch.no_grad():
                    prediction = model(input_batch)
                    depth_map = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1), size=img_cv.shape[:2], mode="bicubic"
                    ).squeeze().cpu().numpy()

                # Results logic
                if mode == "Side-by-Side (Stereopair)":
                    result = make_sbs(img, depth_map)
                    st.subheader("Your 3D Result")
                    st.image(result, use_container_width=True)
                else:
                    # Show Depth Map for Option 2
                    st.subheader("AI Generated Depth Map (The 'Hidden' Shape)")
                    depth_vis = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
                    st.image(depth_vis, caption="This is the 3D outline hidden in the pattern.", width=300)
                    
                    result = make_magic_eye(depth_map)
                    st.subheader("Your Stereogram Result!")
                    st.image(result, use_container_width=True, channels="L")
                
                st.success("Done! You can right-click the image to save it.")