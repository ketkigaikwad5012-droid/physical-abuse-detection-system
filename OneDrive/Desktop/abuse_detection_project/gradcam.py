import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image


# ─────────────────────────────────────────────────────────────────────────────
# LOAD IMAGE  — same preprocessing as training
# ─────────────────────────────────────────────────────────────────────────────
def load_and_preprocess(image_path, img_size=(224, 224)):
    img       = keras_image.load_img(image_path, target_size=img_size)
    img_array = keras_image.img_to_array(img)
    img_array = preprocess_input(img_array)          # ← matches train_model.py
    return np.expand_dims(img_array, axis=0)         # shape (1, 224, 224, 3)


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM  — finds the last conv layer automatically
# ─────────────────────────────────────────────────────────────────────────────
def _get_last_conv_layer(model):
    """Walk layers in reverse and return the first Conv2D found."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # MobileNetV2 wraps its layers inside a sub-model
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return layer.name   # return the wrapper name, not sub-layer
    raise ValueError("No Conv2D layer found in model")


def generate_gradcam(model, img_array, last_conv_layer_name=None):
    """
    Parameters
    ----------
    model               : loaded Keras model
    img_array           : preprocessed image array, shape (1, 224, 224, 3)
    last_conv_layer_name: override auto-detection if needed

    Returns
    -------
    heatmap : float32 ndarray shape (H, W), values 0-1
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = _get_last_conv_layer(model)

    # Build a model that outputs [conv_output, final_prediction]
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # For binary sigmoid: use the output directly
        loss = predictions[:, 0]

    # Gradients of the class score w.r.t. conv feature maps
    grads = tape.gradient(loss, conv_outputs)           # shape (1, h, w, C)

    # Global average pool the gradients → importance weight per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape (C,)

    # Weight the feature maps and average
    conv_outputs = conv_outputs[0]                       # (h, w, C)
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)                   # (h, w)

    # ReLU + normalise to 0-1
    heatmap = tf.nn.relu(heatmap).numpy()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# OVERLAY  — blend heatmap onto the original image
# ─────────────────────────────────────────────────────────────────────────────
def overlay_heatmap(heatmap, original_image_path, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """
    Parameters
    ----------
    heatmap            : float32 ndarray (H, W)
    original_image_path: path to original (un-preprocessed) image
    alpha              : heatmap opacity  (0 = invisible, 1 = full)
    colormap           : OpenCV colormap

    Returns
    -------
    superimposed : BGR ndarray  (same size as original image)
    """
    orig = cv2.imread(original_image_path)
    h, w = orig.shape[:2]

    # Resize heatmap to match original image
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_color   = cv2.applyColorMap(heatmap_uint8, colormap)

    superimposed = cv2.addWeighted(heatmap_color, alpha, orig, 1 - alpha, 0)
    return superimposed


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API  — single call used by predict.py and app.py
# ─────────────────────────────────────────────────────────────────────────────
def run_gradcam(model, image_path):
    """
    Parameters
    ----------
    model      : loaded Keras model
    image_path : str path to image

    Returns
    -------
    dict with keys:
        heatmap           : float32 ndarray (H, W)
        gradcam_image     : BGR ndarray with heatmap overlaid
        explanation_text  : str short human-readable description
    """
    img_array = load_and_preprocess(image_path)
    heatmap   = generate_gradcam(model, img_array)
    overlay   = overlay_heatmap(heatmap, image_path)

    # Describe where the model is looking
    h, w     = heatmap.shape
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

    x_zone = "left"   if peak_x < w / 3 else ("right" if peak_x > 2 * w / 3 else "centre")
    y_zone = "upper"  if peak_y < h / 3 else ("lower" if peak_y > 2 * h / 3 else "middle")

    explanation = (
        f"Model focused on the {y_zone}-{x_zone} region of the image. "
        f"Brighter areas on the heatmap show where the CNN found the most "
        f"evidence for its decision."
    )

    return {
        "heatmap":          heatmap,
        "gradcam_image":    overlay,
        "explanation_text": explanation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST  —  python gradcam.py path/to/image.jpg
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    path       = sys.argv[1] if len(sys.argv) > 1 else input("Image path: ")
    model_path = "models/best_model.keras"

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Train it first with train_model.py")
        exit(1)

    mdl    = load_model(model_path)
    result = run_gradcam(mdl, path)

    print("\n" + result["explanation_text"])
    cv2.imshow("Grad-CAM", result["gradcam_image"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
