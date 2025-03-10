import streamlit as st
import pandas as pd
import base64
from pickle import load
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Arc
# from joblib import dump, loadx
# def draw_speedometer(risk_score):
#     fig, ax = plt.subplots(figsize=(6, 3))
#     ax.set_xlim(-1.2, 1.2)
#     ax.set_ylim(-0.5, 1.2)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_frame_on(False)
    
#     # Draw arc (Speedometer regions)
#     arc = Arc((0, 0), 2, 2, theta1=0, theta2=180, color='black', lw=2)
#     ax.add_patch(arc)
    
#     # Color regions
#     ax.fill_between([-1, -0.3], [0, 1], color='green', alpha=0.6, label='Low Risk')
#     ax.fill_between([-0.3, 0.3], [1, 1], color='yellow', alpha=0.6, label='Moderate Risk')
#     ax.fill_between([0.3, 1], [1, 0], color='red', alpha=0.6, label='High Risk')
    
#     # Animate needle movement
#     min_angle, max_angle = -90, 90
#     if risk_score >= 5:
#         target_angle = 90  # High risk
#     elif 3 <= risk_score < 5:
#         target_angle = 0  # Moderate risk
#     else:
#         target_angle = -90  # Low risk
    
#     for angle in np.linspace(min_angle, target_angle, num=30):
#         ax.clear()
#         ax.set_xlim(-1.2, 1.2)
#         ax.set_ylim(-0.5, 1.2)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_frame_on(False)
        
#         # Redraw arc and regions
#         ax.add_patch(Arc((0, 0), 2, 2, theta1=0, theta2=180, color='black', lw=2))
#         ax.fill_between([-1, -0.3], [0, 1], color='green', alpha=0.6)
#         ax.fill_between([-0.3, 0.3], [1, 1], color='yellow', alpha=0.6)
#         ax.fill_between([0.3, 1], [1, 0], color='red', alpha=0.6)
        
#         # Draw needle
#         x, y = np.cos(np.radians(angle)), np.sin(np.radians(angle))
#         ax.plot([0, x], [0, y], color='black', lw=3)
        
#         st.pyplot(fig)
#         time.sleep(0.05)
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
st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
        box-shadow: 0 0 10px #4CAF50; /* Glowing effect */
        animation: glowing 1.5s infinite;
    }

    @keyframes glowing {
        0% { box-shadow: 0 0 5px #4CAF50; }
        50% { box-shadow: 0 0 20px #4CAF50; }
        100% { box-shadow: 0 0 5px #4CAF50; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Custom CSS for radiating effect
st.markdown(
    """
    <style>
    /* Add a glowing border effect to the sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(90deg, #ff8a8a, #ffc0c0); /* Radiating color gradient */
        border: 2px solid #ff6b6b;
        box-shadow: 0 0 15px #ff6b6b;
    }

    /* Optional: Customize sidebar title */
    [data-testid="stSidebar"] h2 {
        color: #ffffff;
        font-weight: bold;
        text-align: center;
        text-shadow: 2px 2px 5px #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# st.set_page_config(page_title="Web-based Covid Screening System", page_icon="🌟")
# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Risk Assessment", "Descriptive Analysis","Primary Treatment"])

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
        <h1 class="header">{title}</h1>
        """,
        unsafe_allow_html=True,
    )
# header(" Web-based Covid Screening System")
# Set the title and icon for the web app
# Page 1: Risk Assessment
if page == "Risk Assessment":
    add_bg_from_local("content/new_test1.jpg")  # Background for Risk Assessment page
    header("Risk Assessment of COVID-19")
    USER_INPUT = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    st.markdown(
        """
        <h3 style="color: #ff9933;">Enter Details Below for Risk Assessment:</h3>
        """,
        unsafe_allow_html=True,
    )
    
    # Input fields

    name1 = st.text_input("Name")
    AGE = st.number_input("Age", step=1.,format="%.f")
    USER_INPUT[0] = AGE
    gender1 = st.selectbox("Gender", ["Male", "Female", "Other"])
    USER_INPUT[1] = gender1
    pre_medical1 = st.selectbox("Pre-Medical Condition", ["Yes", "No"])
    USER_INPUT[2] = pre_medical1
    if USER_INPUT[2] == 'Yes':
        USER_INPUT[2] = 1
    elif USER_INPUT[2] == 'No':
        USER_INPUT[2] = 0
    E_gene = st.number_input('CT value E gene', step=1.,format="%.f")
    USER_INPUT[3] = E_gene
    # Define the list of symptoms
    symptoms = [
        "Fever", "Cough", "Breathlessness", "Sore Throat", "Loss of Taste/Smell",
        "Body Ache", "Diarrhea", "Vomiting", "Sputum", "Nausea", "Nasal Discharge",
        "Abdominal Pain", "Chest Pain", "Haemoptysis", "Headache", "Body Pain",
        "Weakness", "Cold"
    ]
    
    # Function to collect symptom data
    def collect_symptoms(symptoms):
        """
        Collect symptom data using checkboxes.
        Returns a dictionary with symptom names as keys and their values (1 for selected, 0 for not selected).
        """
        symptom_values = {}
        
        # Split symptoms into two columns for better UI
        col1, col2 = st.columns(2)
        
        for i, symptom in enumerate(symptoms):
            with col1 if i < len(symptoms) // 2 else col2:
                selected = st.checkbox(symptom)
                symptom_values[symptom] = 1 if selected else 0
        
        return symptom_values
    symptom_values = collect_symptoms(symptoms)
    def calculate_risk_score(symptom_values, pre_medical):
        """
        Calculate risk score based on selected symptoms and pre-existing medical conditions.
        """
        risk_score = sum(symptom_values.values())  # Sum of selected symptoms
        if pre_medical == "Yes":
            risk_score += 1  # Add 1 if pre-existing medical condition exists
        return risk_score
        # Step 2: Collect symptom data
    
            # draw_speedometer(risk_score)

            
    # Create two columns for the first two panels
    
    # header("Screening for Covid-19 virus")
    # col1, col2 = st.columns(2)
    
    # ###
    
    # def process_input(input_value):
    #     result = input_value
    #     return result
    
    # # Panel 1: Left panel
    # with col1:
    #       st.markdown(
    #           """
    #           <div style="
    #               background-color: #ffdb4d;
    #               padding: 5px,3px;
    #               border: 1px solid #00FF00;
    #               border-radius: 5px;text-align: center;
    #           ">
    #           <h3 style="color: ##00FF00;">Personal and Clinical Data</h3>
    
    #           </div>
    #           """,
    #           unsafe_allow_html=True,
    #       )
    #       # patient_name =st.text_input('Name')


    #       # pre_medical = st.selectbox('Premedical Condition', ('Yes', 'No'))
          
    #       # gender = st.selectbox('Gender', ('Male', 'Female'))
          
    # # Panel 2: Middle panel
    # with col2:
    #       st.markdown(
    #           """
    #           <div style="
    #               background-color: #ffdb4d;
    #               padding: 5px,3px;
    #               border: 1px solid #00FF00;
    #               border-radius: 5px;text-align: center;
    #           ">
    #           <h3 style="color: ##00FF00;">Symptoms Selection</h3>
    #           </div>
    #           """,
    #           unsafe_allow_html=True,
    #       )
          # symptoms = ['fever', 'cough', 'breathlessness', 'body_ache', 'vomiting', 'sore_throat',
                      # 'diarrhoea', 'sputum', 'nausea', 'nasal_discharge', 'loss_of_taste', 'loss_of_smell',
                      # 'abdominal_pain', 'chest_pain', 'haemoptsis', 'head_ache', 'body_pain', 'weak_ness', 'cold']
    
    #       # Split the symptoms into two columns with 10 rows in each column
    #       symptoms_split = [symptoms[:10], symptoms[10:20], symptoms[20:]]
    
    #       # Create a DataFrame to store symptom values
          # symptom_df = pd.DataFrame(columns=symptoms)
    
    
    #       # Create columns for checkboxes (e.g., 2 columns)
    #       coll1, coll2 = st.columns(2)
    
    #       # Initialize a dictionary to store symptom values
    #       symptom_values = {}
    
    #       # Ensure the loop doesn't exceed the length of the split list
    #       for i in range(len(symptoms)):
    #           with coll1 if i < 10 else coll2:
    #               selected = st.checkbox(symptoms[i])
    #               symptom_values[symptoms[i]] = 1 if selected else 0
    
    #       # Append the symptom values to the DataFrame
    #       symptom_df.loc[len(symptom_df)] = symptom_values
    #              selected = st.checkbox(symptoms_split[2][i])
    #              symptom_df[symptoms_split[2][i]] = [1] if selected else [0]
    
          #USER_INPUT[4] = process_input(SH)

    # new_data = pd.DataFrame({'Age' : USER_INPUT[0], 'E_gene' : USER_INPUT[3], 'Pre_medical' : USER_INPUT[2]}, index = [0])
    #       # Concatenate the two DataFrames vertically
    # combined_df = pd.concat([new_data, symptom_check1], axis=1, ignore_index=True)
    # new_table_df=combined_df
    # new_table_df.columns = list(new_data.columns) + list(symptom_check1.columns)
    def prepare_screening_data(symptom_values, age, e_gene, pre_medical):
        """
        Prepare symptom data for predictive screening.
        Returns a DataFrame with user inputs and symptom values.
        """
        # Create a DataFrame for symptoms
        symptom_df = pd.DataFrame([symptom_values])
    
        # Add other user inputs (age, e_gene, pre_medical)
        user_data = {
            "Age": age,
            "E_gene": e_gene,
            "Pre_medical": pre_medical
        }
        user_df = pd.DataFrame([user_data])
    
        # Combine user data and symptom data
        combined_df = pd.concat([user_df, symptom_df], axis=1)
        return combined_df
    new_table_df = prepare_screening_data(symptom_values, AGE, E_gene, pre_medical1)

    # Panel 3: Full-width panel
    st.markdown(
        """
        <div style="
            background-color: #ff9933;
            padding: 5px;
            border: 1px solid #FFA500;
            border-radius: 5px;text-align: center;
        ">
        <h3 style="color: ##00FF00;">Diagonostic recomendation</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write('Data Overview')
    st.write(new_table_df)
    bayes = load(open('content/bayes.pkl', 'rb'))
    logistic = load(open('content/logistic.pkl', 'rb'))
    # random_tree =load(open('content/random_tree.pkl', 'rb'))
    svm_linear = load(open('content/svm_linear.pkl', 'rb'))
    svm_rbf = load(open('content/svm_rbf.pkl', 'rb'))
    # svm_sigmoid = load('content/svm_sigmoid.joblib')
    svm_sigmoid = load(open('content/svm_sigmoid.pkl', 'rb'))
    # tree = load(open('content/tree.pkl', 'rb'))
    # bayes = load('content/naive_bayes_model.joblib')
    # logistic = load('content/logistic_regression_model.joblib')
    # random_tree =load('content/random_forest_model.joblib')
    # svm_linear = load('content/svm_linear_model.joblib')
    # svm_rbf = load('content/svm_rbf_model.joblib')
    # svm_sigmoid = load('content/svm_sigmoid_model.joblib')
    # # svm_sigmoid = load(open('content/svm_sigmoid.pkl', 'rb'))
    # tree = load('content/decision_tree_model.joblib')
    # # Dropdown menu for model selection
    selected_model = st.selectbox('Select a Model', ['Naive Bayes', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM (Linear)', 'SVM (RBF)', 'SVM(Sigmoid)'])
    prediction=0
    # Perform predictions based on the selected model
    
    CSS = """
        <style>
            .header_pred{
                    text-align: left;
                    font-size: 30px;
                    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
                }
        </style>
    """
    
    HEAD_YES = """
            <h6 class="header_pred" style="color:#affc42"> You Have Covid-19 </h6>
    """
    
    HEAD_NO = """
        <h6 class="header_pred" style="color:#affc42"> You Don't Have Covid-19 </h6>
    """
    combined_df=new_table_df.values 
    if st.button('Make Predictions'):
        st.write("Predicted Results:")
        st.write(combined_df)
        # if selected_model == 'Naive Bayes':
        #     prediction = bayes.predict(combined_df)
        #     # st.write("Predicted Results:")
        #     # st.write(f"Fraction Value: {prediction*100}")
        #     if prediction == 1:
        #         st.markdown(CSS, unsafe_allow_html=True)
        #         st.markdown(HEAD_YES, unsafe_allow_html=True)
        #         st.cache_data.clear()
        #     else:
        #         st.cache_data.clear()
        #         st.markdown(CSS, unsafe_allow_html=True)
        #         st.markdown(HEAD_NO, unsafe_allow_html=True)
        #         st.cache_data.clear()
        # elif selected_model == 'Logistic Regression':
        #     prediction = logistic.predict(combined_df)
        #     # st.write("Predicted Results:")
        #     # st.write(f"Fraction Value: {prediction*100}")
        #     if prediction == 1:
        #         st.markdown(CSS, unsafe_allow_html=True)
        #         st.markdown(HEAD_YES, unsafe_allow_html=True)
        #         st.cache_data.clear()
        #     else:
        #         st.cache_data.clear()
        #         st.markdown(CSS, unsafe_allow_html=True)
        #         st.markdown(HEAD_NO, unsafe_allow_html=True)
        #         st.cache_data.clear()
        # elif selected_model == 'Decision Tree':
        #     prediction = tree.predict(combined_df)
        #     st.write("Predicted Results:")
        #     #st.write(f"Fraction Value: {prediction*100}")
        #     if prediction == 1:
        #         st.markdown(CSS, unsafe_allow_html=True)
        #         st.markdown(HEAD_YES, unsafe_allow_html=True)
        #         st.cache_data.clear()
        #     else:
        #         st.cache_data.clear()
        #         st.markdown(CSS, unsafe_allow_html=True)
        #         st.markdown(HEAD_NO, unsafe_allow_html=True)
        #         st.cache_data.clear()
        # elif selected_model == 'Random Forest':
        #     prediction = random_tree.predict(combined_df)
        #     st.write("Predicted Results:")
        #     #st.write(f"Fraction Value: {prediction*100}")
        #     if prediction == 1:
        #         st.markdown(CSS, unsafe_allow_html=True)
        #         st.markdown(HEAD_YES, unsafe_allow_html=True)
        #         st.cache_data.clear()
        #     else:
        #         st.cache_data.clear()
        #         st.markdown(CSS, unsafe_allow_html=True)
        #         st.markdown(HEAD_NO, unsafe_allow_html=True)
        #         st.cache_data.clear()
        # elif selected_model == 'SVM (Linear)':
        #     prediction = svm_linear.predict(combined_df)
        #     # st.write("Predicted Results:")
        #     # st.write(f"Fraction Value: {prediction*100}")
        #     if prediction == 1:
        #         st.markdown(CSS, unsafe_allow_html=True)
        #         st.markdown(HEAD_YES, unsafe_allow_html=True)
        #         st.cache_data.clear()
        #     else:
        #         st.cache_data.clear()
        #         st.markdown(CSS, unsafe_allow_html=True)
        #         st.markdown(HEAD_NO, unsafe_allow_html=True)
        #         st.cache_data.clear()
        # elif selected_model == 'SVM (RBF)':
        #     prediction = svm_rbf.predict(combined_df)
        #     # st.write("Predicted Results:")
        #     # st.write(f"Fraction Value: {prediction*100}")
        #     if prediction == 1:
        #         st.markdown(CSS, unsafe_allow_html=True)
        #         st.markdown(HEAD_YES, unsafe_allow_html=True)
        #         st.cache_data.clear()
        #     else:
        #         st.cache_data.clear()
        #         st.markdown(CSS, unsafe_allow_html=True)
        #         st.markdown(HEAD_NO, unsafe_allow_html=True)
        #         st.cache_data.clear()
        # elif selected_model == 'SVM (Sigmoid)':
        #     prediction = svm_sigmoid.predict(combined_df)
        #     # st.write("Predicted Results:")
        #     # st.write(f"Fraction Value: {prediction*100}")
        #     if prediction == 1:
        #         st.markdown(CSS, unsafe_allow_html=True)
        #         st.markdown(HEAD_YES, unsafe_allow_html=True)
        #         st.cache_data.clear()
        #     else:
        #         st.cache_data.clear()
        #         st.markdown(CSS, unsafe_allow_html=True)
        #         st.markdown(HEAD_NO, unsafe_allow_html=True)
        #         st.cache_data.clear()
    
    # symptoms1 = [
    #     "Fever", "Cough", "Breathlessness", "Sore Throat",
    #     "Loss of Taste/Smell", "Body Ache", "Diarrhea"
    # ]
    # symptom_check1 = [st.checkbox(symptom) for symptom in symptoms1]
    # Title of the Streamlit app
    # st.title("Multiple bar diagram of covid positive patients with comorbidity and symptoms of COVID-19 over different age groups and different risk labels (infectious nature)")
    
        # Assess risk
    # if st.button("Assess Risk"):
    #     risk_score = sum(symptom_check1) + (1 if pre_medical1 == "Yes" else 0)
    #     if risk_score >= 5:
    #         st.error("High Risk of COVID-19. Consult a healthcare provider immediately.")
    #     elif 3 <= risk_score < 5:
    #         st.warning("Moderate Risk. Self-isolate and monitor symptoms.")
    #     else:
    #         st.success("Low Risk. Continue practicing preventive measures.")
    if st.button("Assess Risk"):
        risk_score = calculate_risk_score(symptom_values, pre_medical)
    
        if risk_score >= 5:
            st.markdown(
                '<div style="background-color: white; color: red; padding: 10px; border: 1px solid red; border-radius: 5px;">'
                "High Risk of COVID-19. Consult a healthcare provider immediately."
                "</div>",
                unsafe_allow_html=True,
            )
            # draw_speedometer(risk_score)
        elif 3 <= risk_score < 5:
            st.markdown(
                '<div style="background-color: white; color: orange; padding: 10px; border: 1px solid orange; border-radius: 5px;">'
                "Moderate Risk. Self-isolate and monitor symptoms."
                "</div>",
                unsafe_allow_html=True,
            )
            # draw_speedometer(risk_score)
        else:
            st.markdown(
                '<div style="background-color: white; color: green; padding: 10px; border: 1px solid green; border-radius: 5px;">'
                "Low Risk. Continue practicing preventive measures."
                "</div>",
                unsafe_allow_html=True,
            )
elif page == "Descriptive Analysis":
  # Adding a graph image (JPG format)
    image_path = "content/Risk_stratification_bar_diagram.jpg"  # Path to your JPG file
    st.image(image_path, caption="Graph Representation",  use_container_width=True)
    
    # Additional UI elements (optional)
    st.write("Multiple bar diagram of covid positive patients with comorbidity and symptoms of COVID-19 over different age groups and different risk labels (infectious nature)")
    
  
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
