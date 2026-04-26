import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

from pose_detection import run_pose_analysis
from gradcam        import run_gradcam

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH    = "models/best_model.keras"
CNN_WEIGHT    = 0.6   # how much the CNN score contributes to final verdict
POSE_WEIGHT   = 0.4   # how much the pose score contributes


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL  (cached so app.py doesn't reload on every prediction)
# ─────────────────────────────────────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at '{MODEL_PATH}'.\n"
                "Run train_model.py first."
            )
        _model = load_model(MODEL_PATH)
    return _model


# ─────────────────────────────────────────────────────────────────────────────
# CNN PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def cnn_predict(image_path):
    """
    Returns raw sigmoid score (float 0-1).
    Score > 0.5 → violence  (assuming violence folder = label 1 during training).
    """
    img       = keras_image.load_img(image_path, target_size=(224, 224))
    arr       = keras_image.img_to_array(img)
    arr       = preprocess_input(arr)              # ← matches train_model.py
    arr       = np.expand_dims(arr, axis=0)
    score     = float(get_model().predict(arr, verbose=0)[0][0])
    return score


# ─────────────────────────────────────────────────────────────────────────────
# FUSION  — combine CNN + pose scores
# ─────────────────────────────────────────────────────────────────────────────
def fuse_scores(cnn_score, pose_score):
    """Weighted average of both scores → final violence probability."""
    return round(CNN_WEIGHT * cnn_score + POSE_WEIGHT * pose_score, 3)


# ─────────────────────────────────────────────────────────────────────────────
# BUILD EXPLANATION TEXT
# ─────────────────────────────────────────────────────────────────────────────
def build_explanation(cnn_score, pose_result, gradcam_result, final_score):
    verdict = "VIOLENT" if final_score >= 0.5 else "NON-VIOLENT"
    conf    = final_score if final_score >= 0.5 else (1 - final_score)

    lines = [
        f"━━━ VERDICT: {verdict}  (confidence {conf*100:.1f}%) ━━━",
        "",
        f"  CNN model score  : {cnn_score*100:.1f}%  (weight {int(CNN_WEIGHT*100)}%)",
        f"  Pose score       : {pose_result['pose_violence_score']*100:.1f}%  (weight {int(POSE_WEIGHT*100)}%)",
        f"  Final fused score: {final_score*100:.1f}%",
        "",
        "── Pose analysis ─────────────────────────────────────",
    ]

    if pose_result["pose_detected"]:
        for flag in pose_result["pose_flags"]:
            lines.append(f"  • {flag}")
    else:
        lines.append("  • Person not detected — pose score set to 0")

    lines += [
        "",
        "── CNN visual explanation ────────────────────────────",
        f"  {gradcam_result['explanation_text']}",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API  — used by app.py
# ─────────────────────────────────────────────────────────────────────────────
def predict(image_path):
    """
    Full pipeline: image → verdict + explanation + visual outputs.

    Returns
    -------
    dict with keys:
        verdict          : "VIOLENT" or "NON-VIOLENT"
        final_score      : float  0-1
        cnn_score        : float  0-1
        pose_score       : float  0-1
        pose_detected    : bool
        pose_flags       : list[str]
        pose_features    : dict
        explanation      : str   (full human-readable report)
        annotated_image  : BGR ndarray  (pose skeleton)
        gradcam_image    : BGR ndarray  (heatmap overlay)
    """
    model = get_model()

    cnn_score    = cnn_predict(image_path)
    pose_result  = run_pose_analysis(image_path)
    gradcam_result = run_gradcam(model, image_path)

    final_score  = fuse_scores(cnn_score, pose_result["pose_violence_score"])
    verdict      = "VIOLENT" if final_score >= 0.5 else "NON-VIOLENT"
    explanation  = build_explanation(cnn_score, pose_result, gradcam_result, final_score)

    return {
        "verdict":         verdict,
        "final_score":     final_score,
        "cnn_score":       round(cnn_score, 3),
        "pose_score":      pose_result["pose_violence_score"],
        "pose_detected":   pose_result["pose_detected"],
        "pose_flags":      pose_result["pose_flags"],
        "pose_features":   pose_result["pose_features"],
        "explanation":     explanation,
        "annotated_image": pose_result["annotated_image"],
        "gradcam_image":   gradcam_result["gradcam_image"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST  —  python predict.py path/to/image.jpg
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, cv2
    path   = sys.argv[1] if len(sys.argv) > 1 else input("Image path: ")
    result = predict(path)

    print("\n" + result["explanation"])

    cv2.imshow("Pose Skeleton", result["annotated_image"])
    cv2.imshow("Grad-CAM",      result["gradcam_image"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
