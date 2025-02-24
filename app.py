import streamlit as st
import pickle
import joblib
import numpy as np

# Load the Neural Network model with error handling
def load_model():
    model_filename = "neural_network_model.pkl"
    try:
        # First, try loading with pickle
        with open(model_filename, "rb") as model_file:
            model = pickle.load(model_file)
        return model
    except (AttributeError, pickle.UnpicklingError):
        st.error("Error loading model! Possible version mismatch.")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        st.stop()

# Load model
loaded_nn_model = load_model()

# Function to collect user input
def collect_user_input():
    st.sidebar.title("User Input")
    age = st.sidebar.slider("Age", min_value=18, max_value=100, value=25)
    year_options = {1: "Biomed1", 2: "Biomed2", 3: "Biomed3", 4: "Mmed1", 5: "Mmed2", 6: "Mmed3"}
    year = st.sidebar.selectbox("Curriculum Year", list(year_options.keys()), format_func=lambda x: year_options[x])
    sex_options = {1: "Man", 2: "Woman", 3: "Non-binary"}
    sex = st.sidebar.selectbox("Gender", list(sex_options.keys()), format_func=lambda x: sex_options[x])
    
    glang_options = {
        1: "French", 15: "German", 20: "English", 37: "Arab", 51: "Basque", 52: "Bulgarian", 53: "Catalan", 
        54: "Chinese", 59: "Korean", 60: "Croatian", 62: "Danish", 63: "Spanish", 82: "Estonian", 83: "Finnish", 
        84: "Galician", 85: "Greek", 86: "Hebrew", 87: "Hindi", 88: "Hungarian", 89: "Indonesian", 90: "Italian", 
        92: "Japanese", 93: "Kazakh", 94: "Latvian", 95: "Lithuanian", 96: "Malay", 98: "Dutch", 100: "Norwegian", 
        101: "Polish", 102: "Portuguese", 104: "Romanian", 106: "Russian", 108: "Serbian", 112: "Slovak", 
        113: "Slovenian", 114: "Swedish", 116: "Czech", 117: "Thai", 118: "Turkish", 119: "Ukrainian", 
        120: "Vietnamese", 121: "Other"
    }
    glang = st.sidebar.selectbox("Mother Tongue", list(glang_options.keys()), format_func=lambda x: glang_options[x])
    
    part = st.sidebar.selectbox("Partnership Status", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    job = st.sidebar.selectbox("Having a Job", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    stud_h = st.sidebar.slider("Average Hours of Study per Week", min_value=0, max_value=50, value=20)
    health = st.sidebar.selectbox("Satisfaction with Health", [1, 2, 3, 4, 5], 
                                  format_func=lambda x: ["Very dissatisfied", "Dissatisfied", "Neither satisfied nor dissatisfied", "Satisfied", "Very satisfied"][x-1])
    psyt = st.sidebar.selectbox("Consulted with Psychotherapy Last Year", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    jspe = st.sidebar.slider("JSPE Total Empathy Score", min_value=0, max_value=100, value=50)
    qcae_cog = st.sidebar.slider("QCAE Cognitive Empathy Score", min_value=0, max_value=100, value=50)
    qcae_aff = st.sidebar.slider("QCAE Affective Empathy Score", min_value=0, max_value=100, value=50)
    asmp = st.sidebar.slider("AMSP Total Score", min_value=0, max_value=100, value=50)
    erec_mean = st.sidebar.slider("GERT Mean Value of Correct Responses", min_value=0, max_value=100, value=50)
    cesd = st.sidebar.slider("CES-D Total Score", min_value=0, max_value=100, value=50)
    stai_t = st.sidebar.slider("STAI Score", min_value=0, max_value=100, value=50)

    return {
        "age": age,
        "year": year,
        "sex": sex,
        "glang": glang,
        "part": part,
        "job": job,
        "stud_h": stud_h,
        "health": health,
        "psyt": psyt,
        "jspe": jspe,
        "qcae_cog": qcae_cog,
        "qcae_aff": qcae_aff,
        "asmp": asmp,
        "erec_mean": erec_mean,
        "cesd": cesd,
        "stai_t": stai_t,
    }

# Collect user input
user_input = collect_user_input()

# Display collected user input
st.title("User Input")
st.write(user_input)

# Function to make prediction
def make_prediction(model, user_input):
    try:
        # Convert input data to NumPy array
        features_for_prediction = np.array([user_input[key] for key in user_input.keys()]).reshape(1, -1)

        # Make prediction
        burnout_prediction = model.predict(features_for_prediction)[0]

        # Map the predicted value to burnout categories
        burnout_category = 'Low or No Burnout' if burnout_prediction == 0 else 'Moderate or High Burnout'

        return burnout_category
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Display the prediction
if st.button("Predict"):
    burnout_category_prediction = make_prediction(loaded_nn_model, user_input)
    if burnout_category_prediction:
        st.title("Prediction for Burnout Category")
        st.write(f"Predicted Burnout Category: {burnout_category_prediction}")
