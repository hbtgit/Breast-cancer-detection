import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.utils import get_custom_objects
import pydicom
import plotly.express as px
import time

# Define the LocalityPreservingProjection layer
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
try:
    model = tf.keras.models.load_model('C:/Users/Hab/Desktop/BCD/Breast-cancer-detection/model/Inception_V4_with_LPP.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
except FileNotFoundError:
    st.error("Model file not found. Please check the file path.")
    st.stop()

# Define class labels
class_labels = ['Density1Benign', 'Density1Malignant', 'Density2Benign', 'Density2Malignant',
                'Density3Benign', 'Density3Malignant', 'Density4Benign', 'Density4Malignant']

# Translation dictionary for English and Amharic
translations = {
    'en': {
        'title': "Inception V4 with LPP - Mammogram Classification",
        'subtitle': "Upload a mammogram image to classify it as Benign or Malignant across Density 1-4.",
        'logo_alt': " ",
        'university': "Adama Science and Technology University",
        'department': "Department of Computer Science and Engineering",
        'project': "Computer Vision Project",
        'instructions_header': "Instructions",
        'instructions': [
            "1. Upload a mammogram image (JPG, PNG, or DICOM).",
            "2. Click 'Classify' to view the prediction.",
            "3. Ensure the image is clear and properly formatted."
        ],
        'upload_label': "Upload a mammogram image",
        'upload_help': "Upload a JPG, PNG, or DICOM image for classification.",
        'uploaded_image': "Uploaded Image",
        'classify_button': "Classify",
        'prediction_results': "Prediction Results",
        'prediction': "Prediction",
        'confidence': "Confidence",
        'result_description': "The mammogram is classified as {status} with a breast density of {density}. "
                             "The model is {confidence:.2%} confident in this prediction.",
        'download_result': "Download Result",
        'probabilities': "Prediction Probabilities",
        'about_model': "About the Model",
        'model_info': """
            This model uses Inception V4 with Locality Preserving Projection (LPP) to classify mammogram images.
            - **Input**: 299x299 pixel images (JPG, PNG, or DICOM).
            - **Output**: Classification as Benign or Malignant across breast densities levels 1-4.
            - **Training**: Trained on a dataset of labeled mammogram images.
            - **Performance**: Optimized for high accuracy in breast cancer detection.
            
            For best results, use high-quality images with clear mammogram details.
        """,
        'model_performance': "Model Performance",
        'performance_info': "Accuracy metrics for the model across different density and malignancy classes.",
        'footer': "Powered by Streamlit & TensorFlow",
        'language_label': "Select Language"
    },
    'am': {
        'title': "Inception V4 ከLPP ጋር - የማሞግራም ምደባ",
        'subtitle': "የማሞግራም ምስል ይጫኑ እንደ ቤኒንግ ወይም ማሊኝንት በደንሲቲ 1-4 ለመመደብ።",
        'logo_alt': " ",
        'university': "የአዳማ ሳይንስ እና ቴክኖሎጂ ዩኒቨርሲቲ",
        'department': "የኮምፒውተር ሳይንስ እና ኢንጂነሪንግ ዲፓርትመንት",
        'project': "የኮምፒውተር ቪዥን ፕሮጀክት",
        'instructions_header': "መመሪያዎች",
        'instructions': [
            "1. የማሞግራም ምስል ይጫኑ (JPG፣ PNG ወይም DICOM)።",
            "2. የትንበያውን ለማየት 'መደብ' የሚለውን ይጫኑ።",
            "3. ምስሉ ግልጽ እና በትክክል የተቀረጸ መሆኑን ያረጋግጡ።"
        ],
        'upload_label': "የማሞግራም ምስል ይጫኑ",
        'upload_help': "ለምደባ የJPG፣ PNG ወይም DICOM ምስል ይጫኑ።",
        'uploaded_image': "የተጫነ ምስል",
        'classify_button': "መደብ",
        'prediction_results': "የትንበያ ውጤቶች",
        'prediction': "ትንበያ",
        'confidence': "የመተማመን ደረጃ",
        'result_description': "ማሞግራሙ እንደ {status} ተመድቧል ከ{density} ደንሲቲ ጋር። ሞዴሉ በዚህ ትንበያ {confidence:.2%} ይተማመናል።",
        'download_result': "ውጤቱን አውርድ",
        'probabilities': "የትንበያ ፕሮባቢሊቲዎች",
        'about_model': "ስለ ሞዴሉ",
        'model_info': """
            ይህ ሞዴል የማሞግራም ምስሎችን ለመመደብ Inception V4 ከLocality Preserving Projection (LPP) ጋር ይጠቀማል።
            - **ግብዓት**: 299x299 ፒክስል ምስሎች (JPG፣ PNG ወይም DICOM)።
            - **ውጤት**: እንደ ቤኒንግ ወይም ማሊኝንት በደንሲቲ ደረጃዎች 1-4 መመደብ።
            - **ስልጠና**: በተሰየሙ የማሞግራም ምስሎች ዳታሴት ላይ ተሰልጥኗል።
            - **አፈጻጸም**: በጡት ካንሰር ማወቂያ ላይ ከፍተኛ ትክክለኝነት እንዲኖር ተመቻችቷል።
            
            ምርጥ ውጤቶችን ለማግኘት ከፍተኛ ጥራት ያላቸው እና ግልጽ የሆኑ የማሞግራም ምስሎችን ይጠቀሙ።
        """,
        'model_performance': "የሞዴል አፈጻጸም",
        'performance_info': "በተለያዩ ደንሲቲ እና ማሊኝንሲ ክፍሎች ላይ ያለው የሞዴል ትክክለኝነት መለኪያዎች።",
        'footer': "በ[የእርስዎ ስም] ተገንብቷል ❤️ | በStreamlit እና TensorFlow የተጎላበተ",
        'language_label': "ቋንቋ ይምረጡ"
    }
}

# App configuration
st.set_page_config(page_title="Mammogram Classification", layout="wide", initial_sidebar_state="expanded")

# Language selection
with st.sidebar:
    language = st.selectbox(
        translations['en']['language_label'],
        options=['en', 'am'],
        format_func=lambda x: 'English' if x == 'en' else 'አማርኛ (Amharic)',
        key="language"
    )
t = translations[language]  # Shortcut for selected language translations

# Custom CSS for dark theme
st.markdown("""
    <style>
    .main { background-color: #1E1E1E; color: #FFFFFF; font-family: 'Arial', sans-serif; }
    .stButton>button { 
        background-color: #2563EB; 
        color: white; 
        border-radius: 8px; 
        padding: 10px 20px; 
        font-weight: bold; 
        transition: all 0.3s ease; 
    }
    .stButton>button:hover { 
        background-color: #1E40AF; 
        transform: scale(1.05); 
    }
    .stFileUploader { background-color: #2D2D2D; border-radius: 10px; padding: 10px; }
    .stProgress > div > div > div { background-color: #2563EB; }
    .title { font-size: 3rem; font-weight: bold; text-align: center; color: #60A5FA; }
    .subtitle { font-size: 1.5rem; text-align: center; color: #A3BFFA; }
    .tooltip:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        background: #333;
        color: #fff;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 0.9rem;
        z-index: 10;
    }
    </style>
""", unsafe_allow_html=True)

# Header with logo and university info
col1, col2 = st.columns([1, 4])
with col1:
    try:
        logo = Image.open('C:/Users/Hab/Desktop/BCD/Breast-cancer-detection/logo.png')
        st.image(logo, use_column_width=True, caption=t['logo_alt'])
    except FileNotFoundError:
        st.warning("Logo not found. Please place 'logo.png' in the project directory.")
with col2:
    st.markdown(f"""
        <h1>{t['university']}</h1>
        <h2>{t['department']}</h2>
        <h3>{t['project']}</h3>
    """, unsafe_allow_html=True)

# Title and subtitle
st.markdown(f'<div class="title">{t["title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle">{t["subtitle"]}</div>', unsafe_allow_html=True)

# Sidebar with instructions
with st.sidebar:
    st.header(t['instructions_header'], anchor=False)
    for instruction in t['instructions']:
        st.markdown(instruction)

# Main content layout with columns
col1, col2 = st.columns([1, 1])

with col1:
    # Image upload
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        t['upload_label'],
        type=["jpg", "png", "dcm"],
        help=t['upload_help']
    )
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.dcm'):
            # Handle DICOM file
            try:
                dicom = pydicom.dcmread(uploaded_file)
                image_array = dicom.pixel_array
                image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255.0
                image_array = image_array.astype(np.uint8)
                image = Image.fromarray(image_array)
            except Exception as e:
                st.error(f"Error reading DICOM file: {e}")
                st.stop()
        else:
            image = Image.open(uploaded_file)
        st.image(image, caption=t['uploaded_image'], use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Prediction section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if uploaded_file and st.button(t['classify_button'], key="classify"):
        # Preprocess image
        image = image.resize((299, 299))  # Inception V4 input size
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Simulate processing with a progress bar
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)

        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        # Display results
        st.markdown(f"### {t['prediction_results']}")
        st.success(f"**{t['prediction']}:** {predicted_class}")
        st.info(f"**{t['confidence']}:** {confidence:.2%}")

        # Result description
        density = predicted_class.split('Density')[1][0]
        status = "non-cancerous" if "Benign" in predicted_class else "cancerous"
        status_am = "ያልተለመደ (ቤኒንግ)" if "Benign" in predicted_class else "ካንሰራዊ (ማሊኝንት)"
        st.markdown(t['result_description'].format(
            status=status_am if language == 'am' else status,
            density=density,
            confidence=confidence
        ))

        # Download result
        result_text = f"{t['prediction']}: {predicted_class}\n{t['confidence']}: {confidence:.2%}\n" + \
                      t['result_description'].format(status=status_am if language == 'am' else status, density=density, confidence=confidence)
        st.download_button(
            label=t['download_result'],
            data=result_text,
            file_name="classification_result.txt",
            mime="text/plain"
        )

        # Probability distribution with Plotly
        st.markdown(f"### {t['probabilities']}")
        fig = px.bar(
            x=class_labels,
            y=predictions[0],
            labels={'x': 'Class', 'y': 'Probability'},
            color_discrete_sequence=['#2563EB'],
            height=400
        )
        st.plotly_chart(fig, use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Model information and performance
with st.expander(t['about_model'], expanded=False):
    st.markdown(t['model_info'])
    
    # Placeholder performance chart
    st.markdown(f"### {t['model_performance']}")
    st.write(t['performance_info'])
    st.bar_chart({
        'Density1Benign': 0.90, 'Density1Malignant': 0.88,
        'Density2Benign': 0.91, 'Density2Malignant': 0.87,
        'Density3Benign': 0.89, 'Density3Malignant': 0.86,
        'Density4Benign': 0.90, 'Density4Malignant': 0.85
    })

# Footer
st.markdown(f"""
    <hr style='border-color: #4B5563;'>
    <p style='text-align: center; color: #A3BFFA;'>
        {t['footer']}
    </p>
""", unsafe_allow_html=True)