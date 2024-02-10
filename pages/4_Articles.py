
import requests
import streamlit as st
from streamlit_lottie import st_lottie
import PIL
import tensorflow as tf


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def load_model():
    try:
        model = tf.keras.applications.InceptionV3(weights='imagenet')
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

def preprocess_image(img):
    img = img.resize((299, 299))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    return img_array

def predict_skin_cancer(image, model):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class = tf.keras.applications.inception_v3.decode_predictions(predictions, top=1)[0][0][1]
    return predicted_class

st.set_page_config(
    page_title="Skin Cancer",
    page_icon="♋",
    layout="wide",
    initial_sidebar_state="expanded",
)

lottie_health = load_lottieurl(
    "https://assets2.lottiefiles.com/packages/lf20_5njp3vgg.json"
)
lottie_welcome = load_lottieurl(
    "https://assets1.lottiefiles.com/packages/lf20_puciaact.json"
)
lottie_healthy = load_lottieurl(
    "https://assets10.lottiefiles.com/packages/lf20_x1gjdldd.json"
)

st.title("Articles about Skin Cancer Detection Through Machine Learning!")
st_lottie(lottie_welcome, height=300, key="welcome")
st.title("Title Advancements in Skin Cancer Detection Through Machine Learning",anchor=None)
st.markdown("""
- Introduction:
Skin cancer is a significant global health concern, with millions of new cases diagnosed each year. Early detection is crucial for effective treatment and improved
patient outcomes.Traditional methods of diagnosing skin cancer rely heavily on visual inspection by dermatologists, which can be subjective and time-consuming.
However, recent advancements in machine learning (ML) have shown promising results in automating the detection and classification of skin lesions, aiding in early
diagnosis and treatment planning.

- Machine Learning in Dermatology:
Machine learning algorithms have demonstrated remarkable capabilities in various medical fields, including dermatology. By analyzing large datasets of images and
clinical data,ML models can learn to distinguish between benign and malignant skin lesions with high accuracy. These algorithms can also assist dermatologists in
making more accurate diagnoses and reducing unnecessary biopsies.

- Types of Skin Cancer Detection:
There are three main types of skin cancer: melanoma, basal cell carcinoma (BCC), and squamous cell carcinoma (SCC). Machine learning techniques have been applied to
detect each of these types,often utilizing different approaches tailored to the unique characteristics of each lesion.

- Melanoma Detection:
Melanoma is the deadliest form of skin cancer, making early detection critical for patient survival. ML models trained on large databases of dermoscopic images have
demonstrated the ability to differentiate melanoma from benign moles with high sensitivity and specificity. These models can analyze various features of skin lesions,
such as asymmetry, border irregularity, color variation, and diameter, to make accurate predictions.

- Basal Cell Carcinoma and Squamous Cell Carcinoma Detection:
Basal cell carcinoma (BCC) and squamous cell carcinoma (SCC) are more common types of skin cancer, with BCC being the most prevalent. ML algorithms have been developed
to detect these types of skin cancer by analyzing clinical images and patient data. These models can identify specific patterns and characteristics associated with BCC
and SCC, enabling early detection and intervention.

- Challenges and Future Directions:
While machine learning holds tremendous promise in skin cancer detection, several challenges remain. One significant challenge is the need for large and diverse
datasets to train robust models capable of generalizing across different patient populations and skin types. Additionally, ensuring the interpretability and
transparency of ML algorithms is crucial for gaining the trust of clinicians and patients.
Future research directions in this field include the development of more advanced ML techniques, such as deep learning, which can automatically learn relevant features
from raw image data without manual feature extraction. Collaborations between computer scientists, dermatologists, and other healthcare professionals are essential for
translating these technological advancements into clinical practice effectively.

- Conclusion:
Machine learning algorithms have emerged as powerful tools for skin cancer detection, offering the potential to improve diagnostic accuracy, reduce healthcare costs,
and save lives.By leveraging vast amounts of data and sophisticated algorithms, ML models can assist dermatologists in identifying suspicious lesions early,
facilitating timely interventions and improving patient outcomes.Continued research and innovation in this field are crucial for realizing the full potential of machine
learning in dermatology and transforming the way skin cancer is diagnosed and managed.""")

# ... (rest of your introduction code)

# Skin Cancer Prediction Section
# st.header("Skin Cancer Detection")
# pic = st.file_uploader(
#     label="Upload a picture",
#     type=["png", "jpg", "jpeg"],
#     accept_multiple_files=False,
#     help="Upload a picture of your skin to get a diagnosis",
# )

# if st.button("Predict"):
#     if pic:
#         st.header("Results")

#         cols = st.columns([1, 2])
#         with cols[0]:
#             st.image(pic, caption=pic.name, use_column_width=True)

#         with cols[1]:
#             model = load_model()

#             if model:
#                 with st.spinner("Predicting..."):
#                     img = PIL.Image.open(pic)
#                     predicted_class = predict_skin_cancer(img, model)

#                     st.write(f"**Prediction:** `{predicted_class}`")
#             else:
#                 st.error("Model could not be loaded. Please check the model file.")

#         st.warning(
#             ":warning: This is not a medical diagnosis. Please consult a doctor for a professional diagnosis."
#         )
#     else:
#         st.error("Please upload an image")
