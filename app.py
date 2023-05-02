import streamlit as st
import pandas as pd
import numpy as np
import pydicom
import tensorflow as tf
from PIL import Image
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


def read_dicom_file(file):
    dicom = pydicom.dcmread(file)
    img = dicom.pixel_array
    img = np.stack((img,) * 3, axis=-1)
    img = Image.fromarray(img)
    return img.resize((512, 512))


# Create encoders
breast_density_encoder = LabelEncoder()
subtlety_encoder = LabelEncoder()
side_encoder = LabelEncoder()
view_encoder = LabelEncoder()

# Fit encoders
breast_density_encoder.fit([1, 2, 3, 4])
subtlety_encoder.fit([0, 1, 2, 3, 4, 5])
side_encoder.fit(['left', 'right'])
view_encoder.fit(['CC', 'MLO'])


def encode_categorical_data(data, encoder):
    encoded_data = encoder.transform([data])
    one_hot_encoded_data = to_categorical(encoded_data, num_classes=len(encoder.classes_))
    return one_hot_encoded_data


st.title("Breast Cancer Report System")

uploaded_file = st.file_uploader("Upload DICOM Image", type="dcm")
if uploaded_file:
    img = read_dicom_file(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Model loading
    model = tf.keras.models.load_model('path/to/your/saved/model')

    # Inputs
    breast_density = st.selectbox("Breast Density", [1, 2, 3, 4])
    side = st.selectbox("Side", ['left', 'right'])
    view_class = st.selectbox("View Class", ['CC', 'MLO'])
    subtlety_class = st.selectbox("Subtlety Class", [0, 1, 2, 3, 4, 5])

    if st.button("Predict"):
        img_array = np.array(img)
        img_array = img_array.reshape(1, 512, 512, 3)

        # One-hot encode the categorical inputs
        breast_density_one_hot = encode_categorical_data(breast_density, breast_density_encoder)
        subtlety_one_hot = encode_categorical_data(subtlety_class, subtlety_encoder)
        side_one_hot = encode_categorical_data(side, side_encoder)
        view_one_hot = encode_categorical_data(view_class, view_encoder)

        # Create the dataset
        input_data = (img_array, side_one_hot, view_one_hot, breast_density_one_hot, subtlety_one_hot)
        input_dataset = tf.data.Dataset.from_tensor_slices(input_data).batch(1)

        # Make prediction
        prediction = model.predict(input_dataset)

        # Display results
        st.write("Pathology: ", prediction[0].argmax())
        st.write("Assessment: ", prediction[1].argmax())
        st.write("Mass Shape: ", prediction[2].argmax())
        st.write("Mass Margin: ", prediction[3].argmax())
