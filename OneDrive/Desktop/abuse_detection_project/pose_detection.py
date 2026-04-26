import cv2
import mediapipe as mp
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE SETUP
# ─────────────────────────────────────────────────────────────────────────────
_mp_pose    = mp.solutions.pose
_mp_drawing = mp.solutions.drawing_utils
_mp_styles  = mp.solutions.drawing_styles

_pose = _mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5,
)

# MediaPipe 33-landmark indices we care about
_KP = {
    "nose":            0,
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_elbow":     13,
    "right_elbow":    14,
    "left_wrist":     15,
    "right_wrist":    16,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
}


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _angle(a, b, c):
    """Angle in degrees at point b in the triangle a-b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    cos     = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — extract raw landmarks from image
# ─────────────────────────────────────────────────────────────────────────────
def extract_keypoints(image_path):
    """
    Parameters
    ----------
    image_path : str

    Returns
    -------
    landmarks      : list of (x, y, z, visibility) for 33 points, or None
    annotated_img  : BGR ndarray with skeleton drawn
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    results = _pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    annotated = img_bgr.copy()
    if not results.pose_landmarks:
        return None, annotated

    _mp_drawing.draw_landmarks(
        annotated,
        results.pose_landmarks,
        _mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=_mp_styles.get_default_pose_landmarks_style(),
    )

    landmarks = [
        (lm.x, lm.y, lm.z, lm.visibility)
        for lm in results.pose_landmarks.landmark
    ]
    return landmarks, annotated


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — compute human-interpretable features
# ─────────────────────────────────────────────────────────────────────────────
def compute_pose_features(landmarks):
    """
    Returns dict of feature_name -> float using normalised (0-1) coordinates.
    """
    def pt(name):
        lm = landmarks[_KP[name]]
        return (lm[0], lm[1])

    ls, rs  = pt("left_shoulder"),  pt("right_shoulder")
    le, re  = pt("left_elbow"),     pt("right_elbow")
    lw, rw  = pt("left_wrist"),     pt("right_wrist")
    lh, rh  = pt("left_hip"),       pt("right_hip")
    lk, rk  = pt("left_knee"),      pt("right_knee")
    la, ra  = pt("left_ankle"),     pt("right_ankle")
    nose    = pt("nose")

    hip_cx = (lh[0] + rh[0]) / 2
    sh_cy  = (ls[1] + rs[1]) / 2

    return {
        # Arms raised above shoulders (y decreases upward in image coords)
        "left_arm_raised":      1.0 if lw[1] < ls[1] else 0.0,
        "right_arm_raised":     1.0 if rw[1] < rs[1] else 0.0,

        # Elbow bend angles
        "left_elbow_angle":     _angle(ls, le, lw),
        "right_elbow_angle":    _angle(rs, re, rw),

        # Wrist distance to nose (punch / slap)
        "left_wrist_to_nose":   _dist(lw, nose),
        "right_wrist_to_nose":  _dist(rw, nose),

        # Lateral body lean
        "body_lean_x":          abs(nose[0] - hip_cx),

        # Forward lean
        "forward_lean_y":       abs(nose[1] - sh_cy),

        # Knee angles (crouching / kicking)
        "left_knee_angle":      _angle(lh, lk, la),
        "right_knee_angle":     _angle(rh, rk, ra),

        # Body proportions
        "shoulder_width":       _dist(ls, rs),
        "hip_width":            _dist(lh, rh),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — rule-based scoring
# ─────────────────────────────────────────────────────────────────────────────
def analyse_pose(landmarks):
    """
    Returns
    -------
    pose_score : float  0.0 – 1.0
    flags      : list[str]  human-readable triggered rules
    """
    if landmarks is None:
        return 0.0, ["No person detected in image — pose analysis skipped"]

    f      = compute_pose_features(landmarks)
    flags  = []
    score  = 0.0

    if f["left_arm_raised"] or f["right_arm_raised"]:
        flags.append("Arm raised above shoulder level")
        score += 0.20

    if f["left_elbow_angle"] < 90 or f["right_elbow_angle"] < 90:
        flags.append("Sharply bent elbow — possible striking pose")
        score += 0.20

    if f["left_wrist_to_nose"] < 0.15 or f["right_wrist_to_nose"] < 0.15:
        flags.append("Wrist close to head — possible punch or slap")
        score += 0.30

    if f["body_lean_x"] > 0.15:
        flags.append("Strong lateral body lean — aggressive lunge posture")
        score += 0.15

    if f["left_knee_angle"] < 100 or f["right_knee_angle"] < 100:
        flags.append("Deep knee bend — crouching or kicking stance")
        score += 0.15

    if not flags:
        flags.append("No violent pose indicators found")

    return round(min(score, 1.0), 3), flags


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API  — single call used by predict.py and app.py
# ─────────────────────────────────────────────────────────────────────────────
def run_pose_analysis(image_path):
    """
    Returns
    -------
    dict with keys:
        pose_detected        : bool
        pose_violence_score  : float
        pose_flags           : list[str]
        pose_features        : dict
        annotated_image      : BGR ndarray
    """
    landmarks, annotated = extract_keypoints(image_path)
    score, flags         = analyse_pose(landmarks)
    features             = compute_pose_features(landmarks) if landmarks else {}

    return {
        "pose_detected":       landmarks is not None,
        "pose_violence_score": score,
        "pose_flags":          flags,
        "pose_features":       features,
        "annotated_image":     annotated,
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST  —  python pose_detection.py path/to/image.jpg
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    path   = sys.argv[1] if len(sys.argv) > 1 else input("Image path: ")
    result = run_pose_analysis(path)

    print(f"\nPose detected       : {result['pose_detected']}")
    print(f"Pose violence score : {result['pose_violence_score']}")
    print("Triggered rules:")
    for flag in result["pose_flags"]:
        print(f"  • {flag}")

    cv2.imshow("Pose", result["annotated_image"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
