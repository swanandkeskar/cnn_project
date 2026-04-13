import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing import image
import numpy as np

# -----------------------------
# FIX quantization_config error
# -----------------------------
class CustomDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)

# -----------------------------
# MODEL PATHS
# -----------------------------
LEAF_MODEL_PATH = "leaf_detecter.h5"
DISEASE_MODEL_PATH = "disease_detecter.h5"

print("Loading models...")

leaf_model = load_model(
    LEAF_MODEL_PATH,
    compile=False,
    custom_objects={"Dense": CustomDense}
)

disease_model = load_model(
    DISEASE_MODEL_PATH,
    compile=False,
    custom_objects={"Dense": CustomDense}
)

print("Models loaded successfully!")
print("Disease model input shape:", disease_model.input_shape)
# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------

# Leaf detection model uses 224x224
def preprocess_leaf(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Disease model uses 128x128
def preprocess_disease(img_path):
    img = image.load_img(img_path, target_size=(128,128))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# DISEASE CLASS NAMES
# -----------------------------
disease_classes = [
    "Disease Free",
    "Leaf Rust",
    "Leaf Spot"
]

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict(img_path):

    print("\nChecking if image is leaf...")

    leaf_img = preprocess_leaf(img_path)
    leaf_pred = leaf_model.predict(leaf_img)

    leaf_score = leaf_pred[0][0]
    print("Leaf model score:", leaf_score)

    # Adjust threshold depending on class order
    if leaf_score > 0.5:
        print("\n❌ Prediction: Not a Mulberry Leaf")
        return

    print("\n🌿 Prediction: Mulberry Leaf Detected")

    print("\nPredicting disease...")

    disease_img = preprocess_disease(img_path)
    disease_pred = disease_model.predict(disease_img)

    disease_index = np.argmax(disease_pred)
    confidence = np.max(disease_pred) * 100

    print("Disease:", disease_classes[disease_index])
    print("Confidence: {:.2f}%".format(confidence))


# -----------------------------
# MAIN PROGRAM
# -----------------------------
if __name__ == "__main__":

    img_path = "ali.png"   # put test image in project folder
    print("Disease model input shape:", disease_model.input_shape)
    predict(img_path)