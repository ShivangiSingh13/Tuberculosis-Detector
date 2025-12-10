import cv2 as cv
import numpy as np
import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = "tb_detection_resnet50.keras"
IMG_SIZE = 256

# Load the trained model
model = load_model(MODEL_PATH)

# Name of last convolutional layer in ResNet50
# This is usually correct for keras.applications.ResNet50
LAST_CONV_LAYER_NAME = "conv5_block3_out"


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=LAST_CONV_LAYER_NAME):
    """
    img_array: preprocessed image batch of shape (1, IMG_SIZE, IMG_SIZE, 3)
    returns: heatmap (IMG_SIZE x IMG_SIZE) with values in [0, 1]
    """

    # Create a model that maps the input image to the activations of the last conv layer
    # and the final predictions
    last_conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    # Compute the gradient of the TB class score with respect to the
    # output feature map of the last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # Assuming output is sigmoid for TB class at index 0
        loss = predictions[:, 0]

    # Gradients of loss w.r.t last conv layer outputs
    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling of gradients over the spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]  # (H, W, C)
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # ReLU: only keep positive values
    heatmap = tf.maximum(heatmap, 0)

    # Normalize to [0, 1]
    if tf.reduce_max(heatmap) != 0:
        heatmap /= tf.reduce_max(heatmap)

    heatmap = heatmap.numpy()
    return heatmap


def overlay_heatmap_on_image(original_rgb, heatmap, alpha=0.4):
    """
    original_rgb: original image (H, W, 3) in RGB
    heatmap: (H, W) values in [0,1]
    returns: RGB image with heatmap overlay
    """

    # Resize heatmap to match original image
    heatmap_resized = cv.resize(heatmap, (original_rgb.shape[1], original_rgb.shape[0]))

    # Convert heatmap to 0-255 uint8
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply color map (JET)
    heatmap_color = cv.applyColorMap(heatmap_uint8, cv.COLORMAP_JET)

    # Convert original image RGB -> BGR for OpenCV blending
    original_bgr = cv.cvtColor(original_rgb, cv.COLOR_RGB2BGR)

    # Overlay heatmap on original
    overlay_bgr = cv.addWeighted(heatmap_color, alpha, original_bgr, 1 - alpha, 0)

    # Convert back to RGB for Gradio display
    overlay_rgb = cv.cvtColor(overlay_bgr, cv.COLOR_BGR2RGB)

    return overlay_rgb


def predict_xray_with_gradcam(image):
    """
    image: comes from Gradio as RGB numpy array (H, W, 3)
    returns: (label_dict, overlay_image)
    """

    # Keep a copy of original resized image for visualization
    original_resized = cv.resize(image, (IMG_SIZE, IMG_SIZE))

    # Convert to grayscale like training
    gray = cv.cvtColor(original_resized, cv.COLOR_RGB2GRAY)

    # Convert grayscale back to RGB (3 channels)
    rgb = cv.cvtColor(gray, cv.COLOR_GRAY2RGB)

    # Normalize
    rgb = rgb.astype("float32") / 255.0

    # Add batch dimension: (1, H, W, 3)
    input_batch = np.expand_dims(rgb, axis=0)

    # Predict TB probability
    prob = model.predict(input_batch)[0][0]

    # Build label dictionary
    result = {
        "Normal": float(1.0 - prob),
        "Tuberculosis": float(prob)
    }

    # Generate Grad-CAM heatmap for TB class
    heatmap = make_gradcam_heatmap(input_batch, model, LAST_CONV_LAYER_NAME)

    # Overlay heatmap on original image
    overlay = overlay_heatmap_on_image(original_resized, heatmap)

    return result, overlay


# Build Gradio interface
demo = gr.Interface(
    fn=predict_xray_with_gradcam,
    inputs=gr.Image(type="numpy", label="Upload Chest X-Ray"),
    outputs=[
        gr.Label(num_top_classes=2, label="Prediction"),
        gr.Image(type="numpy", label="Grad-CAM Heatmap Overlay")
    ],
    title="Tuberculosis Detection with Grad-CAM (ResNet50)",
    description=(
        "Upload a chest X-ray image to classify it as Normal or Tuberculosis. "
        "The Grad-CAM heatmap highlights lung regions that influenced the model's decision."
    )
)

if __name__ == "__main__":
    demo.launch()
