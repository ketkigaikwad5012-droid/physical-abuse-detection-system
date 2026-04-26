import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import base64
import binascii
from werkzeug.utils import secure_filename

app = Flask(__name__)

from ultralytics import YOLO
yolo_pose = YOLO('yolov8n-pose.pt')

def draw_pose_skeleton(img_path):
    img_bgr = cv2.imread(img_path)
    results  = yolo_pose(img_path, verbose=False)
    skeleton_found = False

    KEY_CONNECTIONS = [
        (5,6),(5,7),(7,9),(6,8),(8,10),
        (5,11),(6,12),(11,12),
        (11,13),(13,15),(12,14),(14,16)
    ]

    for result in results:
        if result.keypoints is None:
            continue
        keypoints = result.keypoints.xy.cpu().numpy()
        boxes     = result.boxes

        for person_idx, person_kps in enumerate(keypoints):
            if len(person_kps) == 0:
                continue
            skeleton_found = True

            for s, e in KEY_CONNECTIONS:
                if s < len(person_kps) and e < len(person_kps):
                    x1,y1 = int(person_kps[s][0]),int(person_kps[s][1])
                    x2,y2 = int(person_kps[e][0]),int(person_kps[e][1])
                    if x1>0 and y1>0 and x2>0 and y2>0:
                        cv2.line(img_bgr,(x1,y1),(x2,y2),(0,100,255),3)

            for kp in person_kps:
                x,y = int(kp[0]),int(kp[1])
                if x>0 and y>0:
                    cv2.circle(img_bgr,(x,y),6,(0,255,0),-1)
                    cv2.circle(img_bgr,(x,y),6,(255,255,255),2)

            if boxes is not None and person_idx < len(boxes):
                box = boxes[person_idx].xyxy.cpu().numpy()[0]
                x1,y1,x2,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
                cv2.rectangle(img_bgr,(x1,y1),(x2,y2),(255,100,0),2)

    _, buffer  = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buffer).decode('utf-8'), skeleton_found


def draw_pose_skeleton_frame(frame_bgr):
    drawn = frame_bgr.copy()
    results = yolo_pose(drawn, verbose=False)
    skeleton_found = False

    KEY_CONNECTIONS = [
        (5,6),(5,7),(7,9),(6,8),(8,10),
        (5,11),(6,12),(11,12),
        (11,13),(13,15),(12,14),(14,16)
    ]

    for result in results:
        if result.keypoints is None:
            continue
        keypoints = result.keypoints.xy.cpu().numpy()
        boxes = result.boxes

        for person_idx, person_kps in enumerate(keypoints):
            if len(person_kps) == 0:
                continue
            skeleton_found = True

            for s, e in KEY_CONNECTIONS:
                if s < len(person_kps) and e < len(person_kps):
                    x1, y1 = int(person_kps[s][0]), int(person_kps[s][1])
                    x2, y2 = int(person_kps[e][0]), int(person_kps[e][1])
                    if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                        cv2.line(drawn, (x1, y1), (x2, y2), (0, 100, 255), 3)

            for kp in person_kps:
                x, y = int(kp[0]), int(kp[1])
                if x > 0 and y > 0:
                    cv2.circle(drawn, (x, y), 6, (0, 255, 0), -1)
                    cv2.circle(drawn, (x, y), 6, (255, 255, 255), 2)

            if boxes is not None and person_idx < len(boxes):
                box = boxes[person_idx].xyxy.cpu().numpy()[0]
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cv2.rectangle(drawn, (x1, y1), (x2, y2), (255, 100, 0), 2)

    _, buffer = cv2.imencode('.jpg', drawn)
    return base64.b64encode(buffer).decode('utf-8'), skeleton_found
# ── Config ───────────────────────────────────────────────────
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXT   = {'jpg', 'jpeg', 'png'}
MODEL_PATH    = 'violence_model_v2.h5'
VIDEO_MODEL_PATH = 'violence_video_model_v2.h5'

MODE_THRESHOLDS = {
    'cctv':         0.2,
    'social_media': 0.4
}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ── Load models ───────────────────────────────────────────────
print("Loading image model...")
try:
    model = load_model(MODEL_PATH, compile=False)
except Exception as e:
    print(f"Load failed with error: {e}")
    print("Attempting alternative load method...")
    model = load_model(MODEL_PATH, compile=False, safe_mode=False)
print("Image model loaded!")

print("Loading video model...")
try:
    video_model = load_model(VIDEO_MODEL_PATH, compile=False)
except Exception as e:
    print(f"Load failed with error: {e}")
    print("Attempting alternative load method...")
    video_model = load_model(VIDEO_MODEL_PATH, compile=False, safe_mode=False)
print("Video model loaded!")



# ── Helpers ───────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def predict_image(img_path, threshold=0.5):
    img       = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    pred      = model.predict(img_array, verbose=0)[0][0]
    label     = 'Violence' if pred > threshold else 'Non-Violence'
    confidence = pred if pred > threshold else 1 - pred
    return label, float(confidence), float(pred)


def predict_frame(frame_bgr, threshold=0.5):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (224, 224))
    img_array = np.expand_dims(resized.astype(np.float32), axis=0) / 255.0
    pred = model.predict(img_array, verbose=0)[0][0]
    label = 'Violence' if pred > threshold else 'Non-Violence'
    confidence = pred if pred > threshold else 1 - pred
    return label, float(confidence), float(pred)

def predict_video_clip(frames_list, threshold=0.5):
    frames_resized = []
    for frame in frames_list:
        f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        f = cv2.resize(f, (64, 64))
        f = f.astype(np.float32) / 255.0
        frames_resized.append(f)
    while len(frames_resized) < 8:
        frames_resized.append(frames_resized[-1])
    clip = np.array(frames_resized[:8], dtype=np.float32)
    clip = np.expand_dims(clip, axis=0)
    pred = video_model.predict(clip, verbose=0)[0][0]
    label = 'Violence' if pred > threshold else 'Non-Violence'
    confidence = pred if pred > threshold else 1 - pred
    return label, float(confidence), float(pred)


def blur_image(img_path):
    img       = cv2.imread(img_path)
    blurred   = cv2.GaussianBlur(img, (51,51), 0)
    _, buffer  = cv2.imencode('.jpg', blurred)
    return base64.b64encode(buffer).decode('utf-8')


def blur_frame(frame_bgr):
    blurred = cv2.GaussianBlur(frame_bgr, (51, 51), 0)
    _, buffer = cv2.imencode('.jpg', blurred)
    return base64.b64encode(buffer).decode('utf-8')

def img_to_base64(img_path):
    with open(img_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def frame_to_base64(frame):
    _, buffer  = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    mode = request.form.get('mode', 'social_media')
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    threshold              = MODE_THRESHOLDS.get(mode, 0.5)
    label, confidence, raw = predict_image(filepath, threshold)
    skeleton_img, skel_found = draw_pose_skeleton(filepath)
    blurred_img = blur_image(filepath) if label == 'Violence' else None
    original_img = img_to_base64(filepath)

    return jsonify({
        'label':          label,
        'confidence':     round(confidence * 100, 1),
        'raw_score':      round(raw, 4),
        'mode':           mode,
        'threshold':      threshold,
        'skeleton_img':   skeleton_img,
        'skeleton_found': skel_found,
        'blurred_img':    blurred_img,
        'original_img':   original_img,
        'filename':       filename
    })

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    files     = request.files.getlist('files')
    mode      = request.form.get('mode', 'social_media')
    threshold = MODE_THRESHOLDS.get(mode, 0.5)
    results   = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            label, confidence, raw = predict_image(filepath, threshold)
            original_img = img_to_base64(filepath)
            blurred_img  = blur_image(filepath) if label == 'Violence' else None

            results.append({
                'filename':     filename,
                'label':        label,
                'confidence':   round(confidence * 100, 1),
                'original_img': original_img,
                'blurred_img':  blurred_img
            })

    violence_count     = sum(1 for r in results if r['label'] == 'Violence')
    non_violence_count = len(results) - violence_count

    return jsonify({
        'results':             results,
        'total':               len(results),
        'violence_count':      violence_count,
        'non_violence_count':  non_violence_count
    })

@app.route('/predict_video', methods=['POST'])
def predict_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    video_file = request.files['video']
    mode       = request.form.get('mode', 'social_media')
    threshold  = MODE_THRESHOLDS.get(mode, 0.5)

    if video_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename   = secure_filename(video_file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(video_path)

    cap          = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = round(total_frames / fps, 1) if fps > 0 else 0

    results         = []
    violence_frames = 0
    frame_buffer    = []
    frame_count     = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_buffer.append(frame)

        if len(frame_buffer) == 8:
            label, confidence, raw = predict_video_clip(
                frame_buffer, threshold
            )

            # Draw skeleton on middle frame
            mid_frame      = frame_buffer[4]
            mid_path       = os.path.join(
                app.config['UPLOAD_FOLDER'], f'mid_{frame_count}.jpg'
            )
            cv2.imwrite(mid_path, mid_frame)
            skeleton_img, skeleton_found = draw_pose_skeleton(mid_path)
            blurred_img = blur_image(mid_path) \
                if label == 'Violence' else None

            timestamp = round(frame_count / fps, 1) if fps > 0 else 0

            results.append({
                'frame':          frame_count,
                'timestamp':      timestamp,
                'label':          label,
                'confidence':     round(confidence * 100, 1),
                'skeleton_img':   skeleton_img,
                'blurred_img':    blurred_img,
                'skeleton_found': skeleton_found
            })

            if label == 'Violence':
                violence_frames += 1

            frame_buffer = []

        frame_count += 1
        if len(results) >= 60:
            break

    cap.release()

    total_analyzed      = len(results)
    violence_percentage = round(
        violence_frames / total_analyzed * 100, 1
    ) if total_analyzed > 0 else 0
    overall_verdict = 'Violence' \
        if violence_percentage > 30 else 'Non-Violence'

    return jsonify({
        'results':             results,
        'total_frames':        total_frames,
        'total_analyzed':      total_analyzed,
        'violence_frames':     violence_frames,
        'violence_percentage': violence_percentage,
        'overall_verdict':     overall_verdict,
        'duration':            duration,
        'fps':                 round(fps, 1),
        'mode':                mode
    })


@app.route('/predict_webcam_frame', methods=['POST'])
def predict_webcam_frame():
    data = request.get_json(silent=True) or {}
    frame_data = data.get('frame')
    mode = data.get('mode', 'cctv')
    threshold = MODE_THRESHOLDS.get(mode, 0.5)

    if not frame_data:
        return jsonify({'error': 'No frame data provided'}), 400

    try:
        if ',' in frame_data:
            frame_data = frame_data.split(',', 1)[1]
        decoded = base64.b64decode(frame_data)
    except (ValueError, binascii.Error):
        return jsonify({'error': 'Invalid frame encoding'}), 400

    np_buffer = np.frombuffer(decoded, dtype=np.uint8)
    frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': 'Unable to decode frame'}), 400

    label, confidence, raw = predict_frame(frame, threshold)
    skeleton_img, skeleton_found = draw_pose_skeleton_frame(frame)
    original_img = frame_to_base64(frame)
    blurred_img = blur_frame(frame) if label == 'Violence' else None

    return jsonify({
        'label': label,
        'confidence': round(confidence * 100, 1),
        'raw_score': round(raw, 4),
        'mode': mode,
        'threshold': threshold,
        'skeleton_img': skeleton_img,
        'skeleton_found': skeleton_found,
        'blurred_img': blurred_img,
        'original_img': original_img
    })

if __name__ == '__main__':
    app.run(debug=True)