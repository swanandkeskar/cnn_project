from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Fix for tensorflow quantization error
class CustomDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)

# Load models
leaf_model = load_model(
    "leaf_detecter.h5",
    compile=False,
    custom_objects={"Dense": CustomDense}
)

disease_model = load_model(
    "disease_detecter.h5",
    compile=False,
    custom_objects={"Dense": CustomDense}
)

# Disease classes
classes = ["Disease Free", "Leaf Rust", "Leaf Spot"]

# Load metadata
metadata = pd.read_excel("metadata.xlsx")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------
# Image preprocessing
# -------------------------

def preprocess_leaf(img_path):

    img = image.load_img(img_path, target_size=(224,224))
    img = image.img_to_array(img)
    img = img/255.0
    img = np.expand_dims(img,axis=0)

    return img


def preprocess_disease(img_path):

    img = image.load_img(img_path, target_size=(128,128))
    img = image.img_to_array(img)
    img = img/255.0
    img = np.expand_dims(img,axis=0)

    return img


# -------------------------
# ROUTE
# -------------------------

@app.route("/",methods=["GET","POST"])
def index():

    prediction=None
    disease_data=None
    img_path=None
    leaf_confidence=None
    disease_confidence=None

    if request.method=="POST":

        file=request.files["image"]

        if file:

            img_path=os.path.join(UPLOAD_FOLDER,file.filename)
            file.save(img_path)

            # Leaf prediction
            leaf_img=preprocess_leaf(img_path)
            leaf_pred=leaf_model.predict(leaf_img)

            score=float(leaf_pred[0][0])

            # Leaf probability
            leaf_probability=(1-score)*100
            leaf_confidence=round(leaf_probability,2)

            print("Leaf score:",score)
            print("Leaf probability:",leaf_probability)

            # Leaf decision
            if leaf_probability>50:

                # Run disease model
                disease_img=preprocess_disease(img_path)
                disease_pred=disease_model.predict(disease_img)

                class_index=np.argmax(disease_pred)

                prediction=classes[class_index]

                disease_confidence=round(np.max(disease_pred)*100,2)

                # Healthy leaf
                if prediction=="Disease Free":
                    disease_confidence=0

                # Get metadata
                if prediction!="Disease Free":

                    row=metadata[metadata["Disease"]==prediction]

                    if not row.empty:
                        disease_data=row.iloc[0].to_dict()

            else:

                prediction="Not a Mulberry Leaf"
                disease_confidence=0


    return render_template(
        "index.html",
        prediction=prediction,
        disease_data=disease_data,
        img_path=img_path,
        leaf_confidence=leaf_confidence,
        disease_confidence=disease_confidence
    )


if __name__=="__main__":
    app.run()