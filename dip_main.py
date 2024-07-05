# streamlit run c:\Users\rohan\OneDrive\Documents\code_VS\DIP_Web\dip_main.py
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json

# Load the saved metrics
with open('C:\\Users\\rohan\\OneDrive\\Documents\\code_VS\\DIP_Web\\model_metrics.json', 'r') as f:
    metrics = json.load(f)

best_threshold = metrics['best_threshold']

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("C:\\Users\\rohan\\OneDrive\\Documents\\code_VS\\DIP_Web\\dip.keras")
    file_bytes = np.asarray(bytearray(test_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    resized_image = cv2.resize(image, (128, 128))
    resized_image = resized_image / 255.0
    resized_image = np.expand_dims(resized_image, axis=0)
    predictions = model.predict(resized_image)
    return predictions[0][0]

# Interpret Prediction
def interpret_prediction(prediction, threshold):
    return 'Manipulated' if prediction >= threshold else 'Original'

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Image Manipulation Detection"])

# Main Page
if app_mode == "Home":
    st.header("Image Manipulation Detection")
    image_path = "C:\\Users\\rohan\\OneDrive\\Documents\\code_VS\\DIP_Web\\home_bg.png"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    
    # Problem Statement
    #### Traditional methods of detecting image manipulation often rely on visual inspection, which can be subjective and time-consuming. This system aims to provide a quick and accurate method to detect manipulated images using advanced machine learning techniques.
    """)

elif app_mode == "About":
    st.header("About")
    st.markdown("""
    # About Dataset
    #### The dataset is taken from Kaggle and contains about 12,616 authentic and tampered
    #### images largely based on animals, objects, landscapes, plants, monuments, etc.
    #### Training samples: 10091
    #### Testing samples: 2523

    # Digital image processing techniques used
    #### 1. Image Resizing
    #### 2. Image normalization
    #### 3. Data Augmentation
    #### 4. Batch normalization

    # Model Building:
    #### 1. Mixed Precision Enables mixed precision training to improve performance.
    #### 2. Convolutional layers, followed by batch normalization and max pooling.
    #### 3. A flatten layer to convert the 2D feature maps into 1D feature vectors.
    #### 4. Dense layers with ReLU activation and dropout for regularization.
    #### 5. An output layer with sigmoid activation for binary classification (authentic or forged).
    #### 6. Early Stopping and Learning Rate Reduction: Defines callbacks for early stopping (to prevent overfitting) and learning rate reduction (to fine-tune the learning process).
    #### 7. Trains the model for 50 epochs using the training and validation datasets. 
    """)

elif app_mode == "Image Manipulation Detection":
    st.header("Image Manipulation Detection")
    test_image = st.file_uploader("Choose an Image:")
    if test_image is not None:
        st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            st.write("Our Prediction")
            prediction = model_prediction(test_image)
            predicted_class = interpret_prediction(prediction, best_threshold)
            
            st.success(f"The given image is {predicted_class}.")
