import streamlit as st
import pandas as pd
import base64
# Function to add a background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Risk Assessment", "Primary Treatment"])

# Header function
def header(title):
    st.markdown(
        f"""
        <style>
        .header {{
            color: #fff;
            text-align: left;
            font-size: 40px;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
        }}
        </style>
        <h1 class="header">{Web_based_covid_19_screeing_and_assesment}</h1>
        """,
        unsafe_allow_html=True,
    )

# Page 1: Risk Assessment
if page == "Risk Assessment":
    add_bg_from_local("content/new_test1.jpg")  # Background for Risk Assessment page
    header("Risk Assessment of COVID-19")
    
    st.markdown(
        """
        <h3 style="color: #ff9933;">Enter Details Below for Risk Assessment:</h3>
        """,
        unsafe_allow_html=True,
    )
    
    # Input fields
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=1, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    pre_medical = st.selectbox("Pre-Medical Condition", ["Yes", "No"])
    
    symptoms = [
        "Fever", "Cough", "Breathlessness", "Sore Throat",
        "Loss of Taste/Smell", "Body Ache", "Diarrhea"
    ]
    symptom_check = [st.checkbox(symptom) for symptom in symptoms]
    
    # Assess risk
    if st.button("Assess Risk"):
        risk_score = sum(symptom_check) + (1 if pre_medical == "Yes" else 0)
        if risk_score >= 5:
            st.error("High Risk of COVID-19. Consult a healthcare provider immediately.")
        elif 3 <= risk_score < 5:
            st.warning("Moderate Risk. Self-isolate and monitor symptoms.")
        else:
            st.success("Low Risk. Continue practicing preventive measures.")

# Page 2: Primary Treatment
elif page == "Primary Treatment":
    # add_bg_from_local("content/primary_treatment_bg.jpg")  # Background for Primary Treatment page
    header("Primary Treatment Instructions")

    st.markdown(
        """
        <h3 style="color: #2d00f7;">General Guidelines for COVID-19 Management:</h3>
        <ul>
            <li>Isolate yourself to prevent the spread of infection.</li>
            <li>Stay hydrated and maintain a balanced diet.</li>
            <li>Monitor your symptoms regularly.</li>
            <li>Take over-the-counter medications for fever or pain as advised by your doctor.</li>
        </ul>
        <h3 style="color: #e500a4;">When to Seek Emergency Care:</h3>
        <ul>
            <li>Difficulty breathing or shortness of breath.</li>
            <li>Persistent chest pain or pressure.</li>
            <li>Confusion or inability to stay awake.</li>
            <li>Bluish lips or face.</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )
