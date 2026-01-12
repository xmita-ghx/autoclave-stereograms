# autoclave-stereograms
This browser-based AI application (https://autoclave-stereograms.streamlit.app/) converts standard 2D images into 3D visual experiences called **Stereograms.** The system uses Computer Vision to estimate depth and Image Processing to generate two distinct types of stereoscopic outputs based on that data.

MODE 1: *Stereogram (Side-by-Side):* Creates 3D by duplicating the image and applying a horizontal pixel shift proportional to the depth map. This simulates binocular parallax, tricking the brain into seeing depth when eyes are crossed.

MODE 2: *Autostereogram (Hidden):* Generates an autostereogram by embedding the depth map into a repeating random noise pattern. The hidden 3D shape emerges only when the viewer decouple their eye focus.

---
**What is a stereogram?**

A stereogram is a 2D image designed to trick the brain into perceiving a 3-dimensional scene. It works by exploiting binocular vision: because our eyes are set apart, they usually view objects from two slightly different angles (parallax), which the brain merges to calculate depth.

---
**How to view a stereogram?**
1) Get Close: Put your nose right against the screen. The image should be completely blurry.
2) Stare Through: Relax your eyes. Pretend you are looking through the screen at a wall far behind it. Do not try to focus on the image.
3) Pull Back Slowly: Very slowly move your head away from the screen (about 1 inch per second).
4) Hold the Stare: Keep your eyes "daydreaming" and unfocused. When you reach about 10–12 inches away, the patterns will start to overlap.
5) Wait for the Snap: A 3D shape will suddenly "pop" out. Once you see the blurry outline, stay still, and your eyes will naturally snap the image into sharp focus.

   Check the Depth Map in your app first so your brain knows what shape to look for!
---

### 1. The Core Process

* **Depth Estimation:** When an image is uploaded, the AI analyzes it to create a "Depth Map." This is a grayscale representation where white pixels represent "near" objects and black pixels represent the "background."
* **Mode 1: Side-by-Side (SBS):** The app duplicates the image and horizontally shifts pixels based on the depth map. This creates a **Parallax Effect**, mimicking how our eyes see slightly different perspectives.
* **Mode 2: Magic Eye (Hidden):** The app generates a random noise pattern and "embeds" the depth map into it by repeating the pattern at specific intervals. The user sees a grayscale preview of the depth map before the final result is generated.

---

### 2. Python Libraries Used

* **`Streamlit`:** The web framework used to build the user interface, handle file uploads, and host the site.
* **`PyTorch` (`torch`):** The engine that runs the **MiDaS** (Multiple Image Depth from Any Scene) AI model.
* **`MiDaS` (via Torch Hub):** The specific machine learning model that performs the depth estimation.
* **`OpenCV` (`cv2`):** Used for advanced image manipulation, specifically converting color spaces (RGB to BGR) for the AI model.
* **`NumPy`:** Handles the heavy mathematical pixel shifting and array manipulations.
* **`Pillow` (`PIL`):** Used for basic image opening, saving, and format conversions within the web app.
* **`io`:** A built-in Python library used to handle the image data in memory so users can download their results without saving files to the server.

---

### 3. Key Constraints

* **Input Hardware:** The code is optimized for both CPU and GPU (CUDA) processing.
* **File Integrity:** A strict gatekeeper ensures images are within the **1.5MB to 5MB** range to guarantee enough pixel data for sharp 3D outlines.
* **Outline Fidelity:** We use **Bicubic Interpolation** and a custom **Mask-Filling algorithm** to ensure the 3D result stays strictly aligned with the original image's silhouette.
