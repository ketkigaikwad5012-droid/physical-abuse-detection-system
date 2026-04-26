════════════════════════════════════════════════════════════════
  ABUSE DETECTION SYSTEM  —  README
  Stack: Python · TensorFlow · MediaPipe · Streamlit
════════════════════════════════════════════════════════════════


─────────────────────────────────────────────────────────────────
 WHAT THIS PROJECT DOES
─────────────────────────────────────────────────────────────────
Given a static image the system:
  1. Runs a MobileNetV2 CNN to classify the image as violent / non-violent
  2. Runs MediaPipe Pose to detect body keypoints and check for violent pose patterns
  3. Generates a Grad-CAM heatmap showing WHERE in the image the CNN focused
  4. Fuses CNN score (60%) + pose score (40%) into a final verdict
  5. Outputs a human-readable explanation of every decision made


─────────────────────────────────────────────────────────────────
 PROJECT FILE STRUCTURE
─────────────────────────────────────────────────────────────────
abuse_detection_project/
│
├── dataset/
│   ├── violence/          ← all violent images go here
│   └── non-violence/      ← all non-violent images go here
│
├── models/                ← created automatically by train_model.py
│   ├── best_model.keras
│   └── violence_model_v2.keras
│
├── outputs/               ← created automatically by train_model.py
│   └── training_results.png
│
├── train_model.py         ← Step 3: train the CNN
├── check_labels.py        ← Step 2b: verify dataset labels
├── pose_detection.py      ← MediaPipe pose logic (used by predict.py)
├── gradcam.py             ← Grad-CAM heatmap logic (used by predict.py)
├── predict.py             ← main prediction pipeline
├── app.py                 ← Streamlit web UI
├── requirements.txt       ← Python dependencies
└── README.txt             ← this file


─────────────────────────────────────────────────────────────────
 STEP 1 — INSTALL PYTHON DEPENDENCIES
─────────────────────────────────────────────────────────────────
Open terminal / command prompt in the project folder and run:

    pip install -r requirements.txt

This installs:
  • tensorflow     — CNN training and inference
  • mediapipe      — pose estimation
  • opencv-python  — image reading and Grad-CAM overlay
  • streamlit      — web UI
  • matplotlib     — training graphs
  • scikit-learn   — optional evaluation utilities
  • Pillow         — image loading


─────────────────────────────────────────────────────────────────
 STEP 2 — DOWNLOAD THE DATASET
─────────────────────────────────────────────────────────────────
We use the Roboflow Violence Detection dataset (6,160 real images).

  Option A — Roboflow website (easiest)
  ───────────────────────────────────────
  1. Go to: https://universe.roboflow.com/securityviolence/violence-detection-p4qev
  2. Click  "Download Dataset"
  3. Select format → "Folder"  (NOT YOLO, NOT COCO)
  4. Download and unzip
  5. You will get two folders: violence/ and non-violence/
  6. Place them inside your dataset/ folder:

       dataset/
           violence/
           non-violence/

  Option B — Python (Roboflow SDK)
  ──────────────────────────────────
  pip install roboflow

  from roboflow import Roboflow
  rf = Roboflow(api_key="YOUR_FREE_API_KEY")   # sign up free at roboflow.com
  project = rf.workspace("securityviolence").project("violence-detection-p4qev")
  dataset = project.version(1).download("folder")


─────────────────────────────────────────────────────────────────
 STEP 2b — VERIFY DATASET LABELS (optional but recommended)
─────────────────────────────────────────────────────────────────
    python check_labels.py

Expected output:
    Class labels: {'non-violence': 0, 'violence': 1}

If your folders are named differently, rename them to exactly:
    violence/     and     non-violence/


─────────────────────────────────────────────────────────────────
 STEP 3 — TRAIN THE MODEL
─────────────────────────────────────────────────────────────────
Open train_model.py and update line 16:

    DATASET_PATH = r"C:\path\to\your\dataset"

Then run:

    python train_model.py

This will:
  • Phase 1 (8 epochs):  train only the custom head, base frozen
  • Phase 2 (15 epochs): fine-tune last 30 MobileNetV2 layers
  • Save  models/best_model.keras  and  models/violence_model_v2.keras
  • Save  outputs/training_results.png  (accuracy + loss graphs)

Training time (CPU):  ~30–60 min depending on your machine
Training time (GPU):  ~5–10 min


─────────────────────────────────────────────────────────────────
 STEP 4 — RUN THE WEB APP
─────────────────────────────────────────────────────────────────
    streamlit run app.py

Opens automatically at http://localhost:8501 in your browser.

  • Upload any JPG or PNG image
  • See: Verdict · Score breakdown · Grad-CAM heatmap · Pose skeleton
  • See: Triggered pose rules · Full text report


─────────────────────────────────────────────────────────────────
 QUICK COMMAND-LINE TESTS (no UI needed)
─────────────────────────────────────────────────────────────────
Test pose detection only:
    python pose_detection.py path/to/image.jpg

Test Grad-CAM only:
    python gradcam.py path/to/image.jpg

Test full pipeline (no UI):
    python predict.py path/to/image.jpg


─────────────────────────────────────────────────────────────────
 HOW SCORING WORKS
─────────────────────────────────────────────────────────────────
  CNN score   (0–1)  ×  0.60
+ Pose score  (0–1)  ×  0.40
= Final score (0–1)

  Final score ≥ 0.50  →  VIOLENT
  Final score <  0.50  →  NON-VIOLENT

Pose rules checked:
  • Arm raised above shoulder level            (+0.20)
  • Sharply bent elbow (< 90°)                 (+0.20)
  • Wrist close to head (dist < 0.15)          (+0.30)
  • Strong lateral body lean                   (+0.15)
  • Deep knee bend (< 100°)                    (+0.15)


─────────────────────────────────────────────────────────────────
 COMMON ERRORS AND FIXES
─────────────────────────────────────────────────────────────────
Error: "No trained model found"
  Fix:  Run train_model.py first

Error: "Cannot read image"
  Fix:  Check the image path is correct and file is not corrupted

Error: class labels reversed (violence=0, non-violence=1)
  Fix:  Rename dataset folders so 'non-violence' comes before 'violence'
        alphabetically, OR swap the threshold logic in predict.py

Error: mediapipe install fails on Windows
  Fix:  pip install mediapipe==0.10.3  (try older version)

Error: tensorflow version conflict
  Fix:  Create a virtual environment:
          python -m venv venv
          venv\Scripts\activate       (Windows)
          pip install -r requirements.txt


─────────────────────────────────────────────────────────────────
 VIRTUAL ENVIRONMENT (recommended)
─────────────────────────────────────────────────────────────────
Windows:
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

Mac / Linux:
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt


════════════════════════════════════════════════════════════════
