import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
import sys

model = load_model('tb_detection_resnet50.keras')
img_size = 256

def predict_xray(image_path):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        print("⚠ Image not found:", image_path)
        return
    
    img = cv.resize(img, (img_size, img_size))
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    img = img.astype('float32')/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        print(f"{image_path} -> Tuberculosis Detected 🟥 (prob={pred:.3f})")
    else:
        print(f"{image_path} -> Normal Lungs 🟩 (prob={pred:.3f})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        predict_xray(sys.argv[1])
