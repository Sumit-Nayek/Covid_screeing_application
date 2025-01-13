import streamlit as st
import pandas as pd
import base64
from pickle import load
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
        <h1 class="header">{title}</h1>
        """,
        unsafe_allow_html=True,
    )

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
    AGE = st.number_input("Age",min_value=1, step=1.,format="%.f")
    USER_INPUT[0] = AGE
    gender1 = st.selectbox("Gender", ["Male", "Female", "Other"])
    USER_INPUT[1] = gender1
    pre_medical1 = st.selectbox("Pre-Medical Condition", ["Yes", "No"])
    USER_INPUT[2] = pre_medical1
    if USER_INPUT[2] == 'Yes':
        USER_INPUT[2] = 1
    elif USER_INPUT[2] == 'No':
        USER_INPUT[2] = 0
    symptoms1 = [
        "Fever", "Cough", "Breathlessness", "Sore Throat",
        "Loss of Taste/Smell", "Body Ache", "Diarrhea"
    ]
    symptom_check1 = [st.checkbox(symptom) for symptom in symptoms1]
    
    # Assess risk
    if st.button("Assess Risk"):
        risk_score = sum(symptom_check1) + (1 if pre_medical1 == "Yes" else 0)
        if risk_score >= 5:
            st.error("High Risk of COVID-19. Consult a healthcare provider immediately.")
        elif 3 <= risk_score < 5:
            st.warning("Moderate Risk. Self-isolate and monitor symptoms.")
        else:
            st.success("Low Risk. Continue practicing preventive measures.")
    # Create two columns for the first two panels
    col1, col2 = st.columns(2)
    
    ###
    def process_input(input_value):
        result = input_value
        return result
    
    # Panel 1: Left panel
    with col1:
          st.markdown(
              """
              <div style="
                  background-color: #ffdb4d;
                  padding: 5px,3px;
                  border: 1px solid #00FF00;
                  border-radius: 5px;text-align: center;
              ">
              <h3 style="color: ##00FF00;">Personal and Clinical Data</h3>
    
              </div>
              """,
              unsafe_allow_html=True,
          )
          # patient_name =st.text_input('Name')

          E_gene = st.number_input('CT value E gene', step=1.,format="%.f")
          USER_INPUT[3] = E_gene
          # pre_medical = st.selectbox('Premedical Condition', ('Yes', 'No'))
          
          # gender = st.selectbox('Gender', ('Male', 'Female'))
          
    # Panel 2: Middle panel
    with col2:
          st.markdown(
              """
              <div style="
                  background-color: #ffdb4d;
                  padding: 5px,3px;
                  border: 1px solid #00FF00;
                  border-radius: 5px;text-align: center;
              ">
              <h3 style="color: ##00FF00;">Symptoms Selection</h3>
              </div>
              """,
              unsafe_allow_html=True,
          )
          symptoms = ['fever', 'cough', 'breathlessness', 'body_ache', 'vomiting', 'sore_throat',
                      'diarrhoea', 'sputum', 'nausea', 'nasal_discharge', 'loss_of_taste', 'loss_of_smell',
                      'abdominal_pain', 'chest_pain', 'haemoptsis', 'head_ache', 'body_pain', 'weak_ness', 'cold']
    
          # Split the symptoms into two columns with 10 rows in each column
          symptoms_split = [symptoms[:10], symptoms[10:20], symptoms[20:]]
    
          # Create a DataFrame to store symptom values
          symptom_df = pd.DataFrame(columns=symptoms)
    
    
          # Create columns for checkboxes (e.g., 2 columns)
          coll1, coll2 = st.columns(2)
    
          # Initialize a dictionary to store symptom values
          symptom_values = {}
    
          # Ensure the loop doesn't exceed the length of the split list
          for i in range(len(symptoms)):
              with coll1 if i < 10 else coll2:
                  selected = st.checkbox(symptoms[i])
                  symptom_values[symptoms[i]] = 1 if selected else 0
    
          # Append the symptom values to the DataFrame
          symptom_df.loc[len(symptom_df)] = symptom_values
    #              selected = st.checkbox(symptoms_split[2][i])
    #              symptom_df[symptoms_split[2][i]] = [1] if selected else [0]
    
          #USER_INPUT[4] = process_input(SH)

          new_data = pd.DataFrame({'Age' : USER_INPUT[0], 'E_gene' : USER_INPUT[3], 'Pre_medical' : USER_INPUT[2]}, index = [0])
          # Concatenate the two DataFrames vertically
          combined_df = pd.concat([new_data, symptom_df], axis=1, ignore_index=True)
          new_table_df=combined_df
          new_table_df.columns = list(new_data.columns) + list(symptom_df.columns)
    
    
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
    # bayes = load(open('content/bayes.pkl', 'rb'))
    # logistic = load(open('content/logistic.pkl', 'rb'))
    # random_tree =load(open('content/random_tree.pkl', 'rb'))
    # svm_linear = load(open('content/svm_linear.pkl', 'rb'))
    # svm_rbf = load(open('content/svm_rbf.pkl', 'rb'))
    # svm_sigmoid = load(open('content/svm_sigmoid.pkl', 'rb'))
    # tree = load(open('content/tree.pkl', 'rb'))
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
            <h6 class="header_pred" style="color:#ff5a5f"> You Have Covid-19 </h6>
    """
    
    HEAD_NO = """
        <h6 class="header_pred" style="color:#affc41"> You Don't Have Covid-19 </h6>
    """
    
    if st.button('Make Predictions'):
        st.write("Predicted Results:")
        # if selected_model == 'Naive Bayes':
        #     prediction = bayes.predict(combined_df)
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
        # elif selected_model == 'Logistic Regression':
        #     prediction = logistic.predict(combined_df)
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
        # elif selected_model == 'SVM (RBF)':
        #     prediction = svm_rbf.predict(combined_df)
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
        # elif selected_model == 'SVM (Sigmoid)':
        #     prediction = svm_sigmoid.predict(combined_df)
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
