import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import keras


models = (r"C:\Users\VAIBHAVRAI\PycharmProjects\fashion_mnist\app\model\trained fashion mnist model.h5")
# NOW WE ARE GOING TO LOAD THE MODEL
model = keras.models.load_model(models)

# Define class label for fashion mnist
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# NOW WE ARE GOING TO DEFINE THE FUNCTION TO PREPROCESS THE UPLOADED IMAGE
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((28,28))
    img = img.convert('L')
    img_array = np.array(img)/ 255.0
    img_array = img_array.reshape((1,28,28,1))
    return img_array

#Now we are going to use the streamlit
st.title('Fashion  item classifier')

uploaded_image = st.file_uploader("Upload and image....", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((100, 100))
        st.image(resized_img)

    with col2:
        if st.button('classify'):
            # HERE WHAT WE ARE GOING TO DO IS PROCESSING THE UPLOADED IMAGE
            img_array = preprocess_image(uploaded_image)

            # NOW WHAT WE ARE GOING TO DO IS THE MAKING THE PREDICTION USING THE TRAIN MODEL
            result = model.predict(img_array)

            # NOW WE ARE GOING TO WRITE THE RESULT
            predicted_class = np.argmax(result)
            prediction  = class_names[predicted_class]

            st.success(f'Prediction: {prediction}')