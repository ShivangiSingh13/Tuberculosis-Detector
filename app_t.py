import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
MODEL_PATH = "tb_detection_resnet50.keras"
IMG_SIZE = 256
LAST_CONV_LAYER_NAME = "conv5_block3_out"  # For ResNet50

# Load model once
model = load_model(MODEL_PATH)


# --------------- GRAD-CAM UTILS ----------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=LAST_CONV_LAYER_NAME):
    """
    img_array: preprocessed image batch of shape (1, IMG_SIZE, IMG_SIZE, 3)
    returns: heatmap (IMG_SIZE x IMG_SIZE) with values in [0, 1]
    """
    last_conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]  # single sigmoid output for TB prob

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]  # (H, W, C)
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # ReLU
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val != 0:
        heatmap /= max_val

    return heatmap.numpy()


def overlay_heatmap_on_image(original_rgb, heatmap, alpha=0.4):
    """
    original_rgb: original image (H, W, 3) in RGB
    heatmap: (H, W) values in [0, 1]
    returns: RGB image with heatmap overlay
    """
    h, w, _ = original_rgb.shape
    heatmap_resized = cv2.resize(heatmap, (w, h))

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(heatmap_color, alpha, original_bgr, 1 - alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    return overlay_rgb


# --------------- APP CLASS ----------------
class TBApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tuberculosis Detection - Deep Learning (ResNet50)")
        self.root.geometry("1100x650")
        self.root.configure(bg="#0f172a")  # dark blue-gray

        # To keep image references
        self.original_img_tk = None
        self.gradcam_img_tk = None
        self.current_image_rgb = None

        self.build_ui()

    def build_ui(self):
        # --------- Title Bar ----------
        title_frame = tk.Frame(self.root, bg="#0f172a")
        title_frame.pack(side=tk.TOP, fill=tk.X, pady=(10, 5))

        title_label = tk.Label(
            title_frame,
            text="🩺 Tuberculosis Detection from Chest X-Ray (ResNet50 + Grad-CAM)",
            font=("Segoe UI", 18, "bold"),
            fg="#e5e7eb",
            bg="#0f172a"
        )
        title_label.pack()

        subtitle_label = tk.Label(
            title_frame,
            text="Upload a chest X-ray → Get prediction + explanation heatmap",
            font=("Segoe UI", 11),
            fg="#9ca3af",
            bg="#0f172a"
        )
        subtitle_label.pack(pady=(2, 5))

        # --------- Main Content Area ----------
        main_frame = tk.Frame(self.root, bg="#0f172a")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left: Image Panels
        left_frame = tk.Frame(main_frame, bg="#0b1120")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Original image panel
        orig_label_title = tk.Label(
            left_frame,
            text="Original X-Ray",
            font=("Segoe UI", 12, "bold"),
            fg="#e5e7eb",
            bg="#0b1120"
        )
        orig_label_title.pack(pady=(10, 5))

        self.orig_canvas = tk.Label(left_frame, bg="#020617")
        self.orig_canvas.pack(padx=10, pady=(0, 10), fill=tk.BOTH, expand=True)

        # Grad-CAM panel
        grad_label_title = tk.Label(
            left_frame,
            text="Grad-CAM Heatmap (Model Focus)",
            font=("Segoe UI", 12, "bold"),
            fg="#e5e7eb",
            bg="#0b1120"
        )
        grad_label_title.pack(pady=(10, 5))

        self.grad_canvas = tk.Label(left_frame, bg="#020617")
        self.grad_canvas.pack(padx=10, pady=(0, 10), fill=tk.BOTH, expand=True)

        # Right: Controls & Info
        right_frame = tk.Frame(main_frame, bg="#020617", bd=0, relief=tk.RIDGE)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Styled Frame
        card = tk.Frame(right_frame, bg="#020617")
        card.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # File label
        self.file_label = tk.Label(
            card,
            text="No image selected",
            font=("Segoe UI", 9),
            fg="#9ca3af",
            bg="#020617",
            wraplength=280,
            justify="left"
        )
        self.file_label.pack(pady=(10, 5), anchor="w")

        # Buttons
        btn_style = {"font": ("Segoe UI", 11, "bold"), "fg": "#0f172a", "bg": "#38bdf8",
                     "activebackground": "#0ea5e9", "activeforeground": "#0f172a",
                     "bd": 0, "relief": tk.FLAT, "cursor": "hand2", "padx": 10, "pady": 5}

        browse_btn = tk.Button(
            card,
            text="📂 Browse X-Ray Image",
            command=self.browse_image,
            **btn_style
        )
        browse_btn.pack(pady=(10, 10), fill=tk.X)

        self.predict_btn = tk.Button(
            card,
            text="🔍 Run Prediction",
            command=self.run_prediction,
            state=tk.DISABLED,
            **btn_style
        )
        self.predict_btn.pack(pady=(0, 15), fill=tk.X)

        # Prediction result display
        self.result_label = tk.Label(
            card,
            text="Prediction: -",
            font=("Segoe UI", 13, "bold"),
            fg="#e5e7eb",
            bg="#020617"
        )
        self.result_label.pack(pady=(10, 5), anchor="w")

        self.prob_label = tk.Label(
            card,
            text="TB Probability: -",
            font=("Segoe UI", 11),
            fg="#e5e7eb",
            bg="#020617"
        )
        self.prob_label.pack(pady=(0, 10), anchor="w")

        # Probability bar (simple visual)
        self.progress = ttk.Progressbar(
            card, orient="horizontal", length=250, mode="determinate"
        )
        self.progress.pack(pady=(0, 5))

        note_label = tk.Label(
            card,
            text="Note: This is a research prototype and not a clinical diagnostic tool.",
            font=("Segoe UI", 8),
            fg="#6b7280",
            bg="#020617",
            wraplength=260,
            justify="left"
        )
        note_label.pack(side=tk.BOTTOM, pady=(10, 10))

        # Style for Progressbar
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TProgressbar", troughcolor="#020617", background="#22c55e")

    # ---------- Actions ----------
    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Chest X-Ray Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return

        self.file_label.config(text=file_path)

        # Load image using PIL
        pil_img = Image.open(file_path).convert("RGB")

        # Keep resized copy for display
        display_img = pil_img.resize((400, 300))
        self.current_image_rgb = np.array(pil_img)

        self.original_img_tk = ImageTk.PhotoImage(display_img)
        self.orig_canvas.config(image=self.original_img_tk)

        # Clear Grad-CAM canvas
        self.grad_canvas.config(image="", text="")

        self.result_label.config(text="Prediction: -")
        self.prob_label.config(text="TB Probability: -")
        self.progress["value"] = 0

        self.predict_btn.config(state=tk.NORMAL)

    def run_prediction(self):
        if self.current_image_rgb is None:
            messagebox.showwarning("No Image", "Please select an image first.")
            return

        # Preprocess image for model
        img_resized = cv2.resize(self.current_image_rgb, (IMG_SIZE, IMG_SIZE))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        rgb = rgb.astype("float32") / 255.0
        input_batch = np.expand_dims(rgb, axis=0)

        # Predict
        prob = model.predict(input_batch)[0][0]
        tb_prob = float(prob)
        normal_prob = float(1.0 - prob)

        if tb_prob > 0.5:
            result_text = "Prediction: TUBERCULOSIS 🟥"
            result_color = "#f97316"
        else:
            result_text = "Prediction: NORMAL 🟩"
            result_color = "#22c55e"

        self.result_label.config(text=result_text, fg=result_color)
        self.prob_label.config(
            text=f"TB Probability: {tb_prob * 100:.2f}% | Normal: {normal_prob * 100:.2f}%"
        )

        # Update progress bar (0–100)
        self.progress["value"] = tb_prob * 100

        # Grad-CAM
        try:
            heatmap = make_gradcam_heatmap(input_batch, model, LAST_CONV_LAYER_NAME)
            overlay_rgb = overlay_heatmap_on_image(img_resized, heatmap)

            # Convert overlay to Tk format
            overlay_pil = Image.fromarray(overlay_rgb).resize((400, 300))
            self.gradcam_img_tk = ImageTk.PhotoImage(overlay_pil)
            self.grad_canvas.config(image=self.gradcam_img_tk)
        except Exception as e:
            print("Grad-CAM error:", e)
            self.grad_canvas.config(
                text="Grad-CAM could not be generated.",
                fg="#f97316",
                bg="#020617",
                font=("Segoe UI", 10, "italic")
            )


# --------------- MAIN ---------------
if __name__ == "__main__":
    root = tk.Tk()
    app = TBApp(root)
    root.mainloop()
