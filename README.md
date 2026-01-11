# autoclave-stereograms
A browser-based AI application that converts standard 2D images into 3D visual experiences. The system uses Computer Vision to estimate depth and Image Processing to generate two distinct types of stereoscopic outputs based on that data. MODE 1: Duplicates and shifts pixels for parallax. MODE 2: Embeds depth map into repeating noise patterns.

---

### 1. The Core Process

* **Depth Estimation:** When an image (strictly between 1MB and 5MB) is uploaded, the AI analyzes it to create a "Depth Map." This is a grayscale representation where white pixels represent "near" objects and black pixels represent the "background."
* **Mode 1: Side-by-Side (SBS):** The app duplicates the image and horizontally shifts pixels based on the depth map. This creates a **Parallax Effect**, mimicking how our eyes see slightly different perspectives.
* **Mode 2: Magic Eye:** The app generates a random noise pattern and "embeds" the depth map into it by repeating the pattern at specific intervals. The user sees a grayscale preview of the depth map before the final result is generated.

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
* **File Integrity:** A strict gatekeeper ensures images are within the **1MB to 5MB** range to guarantee enough pixel data for sharp 3D outlines.
* **Outline Fidelity:** We use **Bicubic Interpolation** and a custom **Mask-Filling algorithm** to ensure the 3D result stays strictly aligned with the original image's silhouette.
