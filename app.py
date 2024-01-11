import streamlit as st
import pickle
import numpy as np

# Function to collect user input
def collect_user_input():
    st.sidebar.title("User Input")
    age = st.sidebar.slider("Age", min_value=18, max_value=100, value=25)
    year_options = {1: "Biomed1", 2: "Biomed2", 3: "Biomed3", 4: "Mmed1", 5: "Mmed2", 6: "Mmed3"}
    year = st.sidebar.selectbox("Curriculum Year", list(year_options.keys()), format_func=lambda x: year_options[x])
    sex_options = {1: "Man", 2: "Woman", 3: "Non-binary"}
    sex = st.sidebar.selectbox("Gender", list(sex_options.keys()), format_func=lambda x: sex_options[x])
    glang_options = {
        1: "French", 15: "German", 20: "English", 37: "Arab", 51: "Basque", 52: "Bulgarian", 53: "Catalan", 54: "Chinese",
        59: "Korean", 60: "Croatian", 62: "Danish", 63: "Spanish", 82: "Estonian", 83: "Finnish", 84: "Galician",
        85: "Greek", 86: "Hebrew", 87: "Hindi", 88: "Hungarian", 89: "Indonesian", 90: "Italian", 92: "Japanese",
        93: "Kazakh", 94: "Latvian", 95: "Lithuanian", 96: "Malay", 98: "Dutch", 100: "Norwegian", 101: "Polish",
        102: "Portuguese", 104: "Romanian", 106: "Russian", 108: "Serbian", 112: "Slovak", 113: "Slovenian",
        114: "Swedish", 116: "Czech", 117: "Thai", 118: "Turkish", 119: "Ukrainian", 120: "Vietnamese", 121: "Other"
    }
    glang = st.sidebar.selectbox("Mother Tongue", list(glang_options.keys()), format_func=lambda x: glang_options[x])
    part_options = {0: "No", 1: "Yes"}
    part = st.sidebar.selectbox("Partnership Status", list(part_options.keys()), format_func=lambda x: part_options[x])
    job_options = {0: "No", 1: "Yes"}
    job = st.sidebar.selectbox("Having a Job", list(job_options.keys()), format_func=lambda x: job_options[x])
    stud_h = st.sidebar.slider("Average Hours of Study per Week", min_value=0, max_value=50, value=20)
    health_options = {1: "Very dissatisfied", 2: "Dissatisfied", 3: "Neither satisfied nor dissatisfied", 4: "Satisfied", 5: "Very satisfied"}
    health = st.sidebar.selectbox("Satisfaction with Health", list(health_options.keys()), format_func=lambda x: health_options[x])
    psyt_options = {0: "No", 1: "Yes"}
    psyt = st.sidebar.selectbox("Consulted with Psychotherapy Last Year", list(psyt_options.keys()), format_func=lambda x: psyt_options[x])
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

# Display the collected user input
st.title("User Input")
st.write(user_input)

# Load the Neural Network model
with open("neural_network_model.pkl", "rb") as model_file:
    loaded_nn_model = pickle.load(model_file)

def make_prediction(model, user_input):
    
    features_for_prediction = [user_input[key] for key in user_input.keys() if key != 'burnout']

    # Make prediction for 'burnout'
    burnout_prediction = model.predict([features_for_prediction])[0]

    # Map the predicted value to burnout categories
    burnout_category = 'Low or No Burnout' if burnout_prediction == 0 else 'Moderate or High Burnout'

    return burnout_category

# Display the prediction for 'burnout'
if st.button("Predict"):
    burnout_category_prediction = make_prediction(loaded_nn_model, user_input)
    st.title("Prediction for Burnout Category")
    st.write(f"Predicted Burnout Category: {burnout_category_prediction}")
