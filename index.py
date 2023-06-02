# import pandas as pd
import cv2
import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
from PIL import Image
import numpy as np
import tensorflow as tf
import keras
from keras.utils import load_img, img_to_array
# from keras.preprocessing import image
from random import randrange
from keras.preprocessing.image import ImageDataGenerator
import pickle
from keras.models import load_model
# import face_recognition
loaded_model = load_model('face_model.h5')

train_dir = "sample_data/Original_Images/"
generator = ImageDataGenerator()
train_ds = generator.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32)
classes = list(train_ds.class_indices.keys())

cap = cv2.VideoCapture(0)
FRAME_WINDOW = st.image([])
text = st.empty()


def predict_image(images_pred, file_name):
    img = images_pred
    actual_name = file_name.split('_')
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    pred = loaded_model.predict(images)
    st.text("Actual: "+actual_name[0])
    st.text("Predicted: "+classes[np.argmax(pred)])


def realtime_predict_image(realtime_images_pred):
    img = realtime_images_pred
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    pred = loaded_model.predict(images)
    predicted = classes[np.argmax(pred)]
    return predicted


def main():
    st.title("Face Recognition App")
    activities = ["Recognition", "Realtime"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'Recognition':
        image_file = st.file_uploader(
            "Upload Image", type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            st.text("Original Image")
            load_image = st.image(image_file)
            our_image = load_img(image_file, target_size=(224, 224, 3))
            # st.image(our_image)
            file_name = image_file.name

        if st.button("Recognize"):
            predict_image(our_image, file_name)

    if choice == 'Realtime':
        st.subheader("Realtime Face Recognition")
        run = st.checkbox("Run camera")  # checkbox
        if run == True:
            while True:
                success, img = cap.read()
                imgS = cv2.resize(img, (224, 224))
                imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(img)
                text.write(realtime_predict_image(imgS))
                cv2.waitKey(1)
        else:
            pass


main()
