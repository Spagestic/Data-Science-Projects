import streamlit as st
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf

# --- Streamlit App ---
st.title('Digit Recognizer App')

st.write('''
This app predicts the digit written by the user using a trained CNN model.\n
Draw a digit in the canvas to see the prediction.
''')

# Create a canvas component
canvas_result = st_canvas(
    fill_color="black", stroke_width=28, stroke_color="white", background_color="black",
    update_streamlit=True, height=280, width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Load the trained CNN model
model_path = 'Kaggle/digit-recognizer/models/cnn_model.h5'
model = tf.keras.models.load_model(model_path)

# If the user has drawn on the canvas, make the prediction
if canvas_result.image_data is not None:
    # Check if the image has 4 channels (RGBA), then convert it to 3 channels (RGB)
    if canvas_result.image_data.shape[-1] == 4:
        # Discard the alpha channel
        img = canvas_result.image_data[:, :, :3]
    else:
        img = canvas_result.image_data

    # Ensure the image data is now in the correct shape [height, width, 3] for RGB images
    if img.shape[-1] != 3:
        print("The input image does not have 3 channels. Please provide an RGB image.")
        print("The input image has shape", img.shape)
    else:
        # Resize the image to 28x28 pixels
        img = tf.image.resize(img, [28, 28])  # Resize to 28x28
        img = tf.image.rgb_to_grayscale(img)  # Convert to grayscale
        img = tf.expand_dims(img, axis=-1)  # Add the channel dimension to make it [28, 28, 1]
        img = tf.expand_dims(img, axis=0)  # Add the batch dimension to make it [1, 28, 28, 1]

        # Now, img is ready for prediction
        prediction = model.predict(img)
        digit = prediction.argmax()

        # Display the prediction
        st.write(f'## Prediction: {digit}')


# --- About Me & Links Section ---
st.markdown("---")
st.header("About the Creator")
st.write("This app was created by Spagestic. You can connect with me on:")

# Replace with your actual links
st.write(f"[GitHub](https://github.com/spagestic)  |  [LinkedIn](https://linkedin.com/in/vishalginni)") 

st.write("Check out the source code for this app on [GitHub](https://github.com/Spagestic/Data-Science-Projects/tree/main/Kaggle/digit-recognizer).") 