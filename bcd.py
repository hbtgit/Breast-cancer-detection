import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.utils import get_custom_objects

# Define the LocalityPreservingProjection layer (replace with actual implementation)
class LocalityPreservingProjection(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(LocalityPreservingProjection, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(LocalityPreservingProjection, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

    def get_config(self):
        config = super(LocalityPreservingProjection, self).get_config()
        config.update({'output_dim': self.output_dim})
        return config

# Register the custom layer
get_custom_objects().update({'LocalityPreservingProjection': LocalityPreservingProjection})

# Load the model
model = tf.keras.models.load_model('/home/hab/BCD/Breast_cancer_detection/model/Inception_V4_with_LPP.h5')

# Compile the model to include metrics (use the same metrics as during training)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define class labels
class_labels = ['Density1Benign', 'Density1Malignant', 'Density2Benign', 'Density2Malignant',
                'Density3Benign', 'Density3Malignant', 'Density4Benign', 'Density4Malignant']

# Apply custom CSS for light blue theme
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background-color: #E6F0FA; /* Light blue background */
        color: #1E3A8A; /* Dark blue text */
    }
    /* Header and title */
    h1, h2, h3 {
        color: #1E3A8A; /* Dark blue for headers */
    }
    /* Buttons */
    .stButton > button {
        background-color: #3B82F6; /* Blue button */
        color: #000000; /* Black text */
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #2563EB; /* Darker blue on hover */
    }
    /* File uploader */
    .stFileUploader {
        background-color: #BFDBFE; /* Lighter blue for uploader */
        border: 1px solid #3B82F6;
        border-radius: 8px;
    }
    /* Success message */
    .stSuccess {
        background-color: #BFDBFE;
        color: #000000;
    }
    /* Spinner */
    .stSpinner > div {
        color: #3B82F6;
    }
    /* Sidebar (if used later) */
    .sidebar .sidebar-content {
        background-color: #BFDBFE;
    }
       /* Ensure all text is black */
    p, div, span, label {
        color: #1E3A8A !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display logo and text side by side
col1, col2 = st.columns([1, 4])  # Adjust ratio for layout
with col1:
    try:
        logo = Image.open('/home/hab/BCD/Breast_cancer_detection/logo.png')
        st.image(logo, width=150)  # Reduced width for balance
    except FileNotFoundError:
        st.warning("Logo not found. Please place 'logo.png' in the project directory.")
with col2:
    st.markdown(
        """
        # Adama Science and Technology University
        """,
        unsafe_allow_html=True
    )
st.markdown(
    """
    ## Department of Computer Science and Engineering
    ### Computer Vision Project
    """,
    unsafe_allow_html=True
)
# Streamlit interface
st.title("Inception V4 with LPP - Mammogram Classification")
st.write("Upload an image to classify it as Benign or Malignant across Density 1-4.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    image = image.resize((299, 299))  # Inception V4 input size
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    with st.spinner("Classifying..."):
        predictions = model.predict(image_array)
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = float(np.max(predictions))

    # Display results
    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2%}")

        # Add description of the result
    density = predicted_class.split('Density')[1][0]  # Extract density number
    status = "non-cancerous" if "Benign" in predicted_class else "cancerous"
    st.write(f"The mammogram is classified as {status} with a breast density of {density}. "
             f"The model is {confidence:.2%} confident in this prediction.")


    # Display probability distribution
    st.subheader("Prediction Probabilities")
    for label, prob in zip(class_labels, predictions[0]):
        st.write(f"{label}: {prob:.2%}")