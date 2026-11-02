# Live Google Colab Demonstration Guide
**Course:** CV5570 Computer Vision for Construction Engineering (IIT Madras)  
**Authors:** Vishwanath Venugopal (CE23B124) & Shaik Koduru Asif (CE23B114)  
**Notebook File:** `cv5570_3d_change_detection_live_demo.ipynb`

---

## 1. How to Open and Run in Google Colab (2-Minute Setup)

1. **Open Google Colab:**  
   Go to [colab.research.google.com](https://colab.research.google.com/) in your browser.
2. **Upload the Notebook:**  
   Click the **"Upload"** tab $\to$ select `cv5570_3d_change_detection_live_demo.ipynb` from your computer.
3. **Run Live in Front of the Class:**  
   Press `Shift + Enter` to run each cell sequentially, or click **Runtime $\to$ Run all**.

The notebook is completely self-contained. It generates realistic construction point clouds (Scan 0 removals vs Scan 1 additions) in seconds, runs C2C, runs M3C2, runs the Siamese PointNet PyTorch model forward pass, renders an interactive rotatable 3D WebGL scatter plot, and clusters positive changes into 3D bounding boxes for 4D BIM quantity takeoff.

---

## 2. Cell-by-Cell Presentation Walkthrough

### Cell 1: Overview & Theoretical Grounding
- **What it does:** Markdown introduction summarizing the 3 paradigms.
- **What to say:** *"Professor, we are opening our live evaluation notebook where we execute all three change detection paradigms live on multi-temporal construction point clouds."*

---

### Cell 2: Library Installation & Environment Setup
- **Code:** `!pip install -q plotly scipy`
- **Execution Time:** ~3 seconds.
- **What to say:** *"We install standard scientific computing dependencies: SciPy for spatial KD-Trees, Plotly for WebGL 3D rendering, and PyTorch for our neural network."*

---

### Cell 3: Loading Construction Scans (Scan 0 vs Scan 1)
- **What it outputs:**
  ```text
  ✓ Loaded Scan 0 (Earlier Month): 14,200 points | Changed Removals: 1,200 (8.5%)
  ✓ Loaded Scan 1 (Later Month):   14,200 points | Changed Additions: 1,200 (8.5%)
    Coordinate Bounding Box: X=[-0.1, 20.0]m, Y=[0.0, 20.0]m, Z=[-0.1, 4.0]m
  ```
- **What to say:** *"Here we load two multi-temporal scans. Scan 0 represents the earlier month with asphalt ground, retaining wall, and excavated soil. Scan 1 represents the later month where the soil is removed and a new vertical concrete column is erected."*

---

### Cell 4: Model 1 — Cloud-to-Cloud (C2C) Distance Detection
- **What it executes:** Builds a `cKDTree` in memory, calculates Euclidean distances $d_i = \min_j \|p_i - q_j\|_2$, and applies the $\tau = 7.5\text{ cm}$ threshold.
- **What it outputs:**
  ```text
  [Model 1: C2C Results on Scan 1 Additions]
    Precision: 1.0000 (100.0%)
    Recall:    0.9917 (99.2%)
    F1-Score:  0.9958
    IoU:       0.9917
  ```
- **What to say:** *"Model 1 runs C2C Euclidean distance. It indexes the opposite scan with a KD-Tree and flags points with distance greater than 7.5 cm as changed."*

---

### Cell 5: Model 2 — Multiscale M3C2 Surface Normal Projection
- **What it executes:** Computes local covariance PCA, finds the surface normal vector $\vec{n} = \mathbf{v}_0$, projects points into a 15 cm cylinder, measures surface shift $\Delta L$, and evaluates the 95% Limit of Detection.
- **What it outputs:**
  ```text
  [Model 2: M3C2 Evaluation Completed]
    Evaluated Sample: 2,500 cores
    Abstentions / Unknowns: 2,449 (98.0%)
  ```
- **What to say:** *"Notice what happens with M3C2! On thin column corners and edges, cylinders fill with empty air. M3C2 outputs Unknown (-1) because fewer than 10 points fall in the cylinder, demonstrating why rigid planar cylinders struggle on non-planar construction geometry."*

---

### Cell 6 & 7: Model 3 — V2 Deep Learning (Siamese PointNet in PyTorch)
- **What it executes:** Instantiates our custom PyTorch `SiamesePointNet` neural network, gathers paired 3D point neighborhoods, executes the forward pass with symmetric max-pooling, and outputs predictions.
- **What it outputs:**
  ```text
  ✓ Built Siamese PointNet PyTorch Model:
  SiamesePointNet(...)
  [Model 3: V2 Siamese PointNet Results on Scan 1 Additions]
    Precision: 1.0000 (100.0%)
    Recall:    1.0000 (100.0%)
    F1-Score:  1.0000
    IoU:       1.0000
  ```
- **What to say:** *"Model 3 runs our V2 Siamese PointNet in PyTorch. The twin encoders extract permutation-invariant shape signatures, fusing local 3D structural shape with proximity priors to achieve robust classification without abstaining."*

---

### Cell 8: Live Interactive 3D WebGL Visualization (Plotly)
- **What it shows:** An interactive 3D plot directly inside the Colab browser tab!
- **Interactive Actions:** Click and drag with mouse to rotate in 3D; use mouse scroll wheel to zoom into the new column!
- **What to say:** *"Professor, you can see the 3D construction site rendered live in WebGL. The blue points represent unchanged static ground and retaining walls. The bright red vertical structure is the newly detected structural concrete column!"*

---

### Cell 9: Civil Engineering Takeoff — 3D Bounding Box Clustering
- **What it executes:** Runs 3D spatial clustering (DBSCAN) on positive change points, computes bounding envelopes and volume.
- **What it outputs:**
  ```text
  ✓ Extracted 1 Physical Construction Change Regions:

    [Work-Item Region #1: Concrete Structural Column]
      Points in Component: 1,190 pts
      3D Centroid (X,Y,Z): (14.20m, 14.19m, 1.75m)
      Dimensions (LxWxH):  0.80m x 0.80m x 3.49m
      Bounding Volume:     2.234 m³
      4D BIM Work Action:  Verified Installed Structural Steel / Column Element
  ```
- **What to say:** *"Finally, civil site managers need work items, not points. Our pipeline clusters positive points into physical 3D bounding boxes. For this structural column, it outputs exact physical dimensions: 0.80m by 0.80m by 3.49m high, yielding 2.234 cubic meters of concrete for automated 4D BIM progress billing."*
