import streamlit as st
import pickle
import pandas as pd
from pickle import load
import base64

def add_bg_from_local(image_file):
    st.write('<style>div.block-container{padding-top:0rem;}</style>', unsafe_allow_html=True)
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
        unsafe_allow_html=True
    )

add_bg_from_local('background/test2.jpg')
def header():
    custom_css = """
        <style>
             .header{
                color: #fff;
                text-align: left;
                font-size: 39px;
                font-weight: bold;

    }
        </style>
    """

    head = """
        <h2 class="header">
            <font color="#2d00f7">Web-enabled Diagonosis for COVID-19</font>
        </h2>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown(head, unsafe_allow_html=True)
header()
# Create two columns for the first two panels
col1, col2 = st.columns(2)
USER_INPUT = [0, 0, 0, 0, 0, 0, 0, 0, 0]
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
          <h4 style="color: ##00FF00;">Patient Information</h4>

          </div>
          """,
          unsafe_allow_html=True,
      )
      patient_name =st.text_input("**Patient's Name**")
      AGE = st.number_input("**Patient\'s Age**", format="%.f")
      USER_INPUT[0] = process_input(AGE)
      gender = st.selectbox("**Patient's Gender**", ('Male', 'Female'))
      USER_INPUT[3] = process_input(gender)
      state=st.text_input('**State**')
      country=st.text_input("**Country**")
      exposed=st.selectbox("**Exposed to covid infected zone**", ('Yes', 'No'))
      no_of_infected_person=st.text_input("**Number of effected person in familiy**")
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
          <h4 style="color: ##00FF00;">Clinical Information</h4>
          </div>
          """,
          unsafe_allow_html=True,
      )
      symptoms = ['Fever', 'Cough', 'Loss of smell', 'Loss of taste','Body ache', 'Vomiting', 'Sore throat',
                  'Diarrhoea', 'Sputum', 'Nausea', 'Chest pain','Breathlessness','Nasal discharge',
                  'Abdominal pain', 'Haemoptsis', 'Head ache', 'Body pain', 'Cold',"Weakness"]

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
      pre_medical = st.selectbox("**Comorbidity ?**", ('Yes', 'No'))
      USER_INPUT[2] = process_input(pre_medical)
      E_gene = st.number_input("**RTPCR test (CT value)**", step=1.,format="%.f")
      USER_INPUT[1] = process_input(E_gene)

      # Append the symptom values to the DataFrame
      symptom_df = symptom_df.append(symptom_values, ignore_index=True)
#              selected = st.checkbox(symptoms_split[2][i])
#              symptom_df[symptoms_split[2][i]] = [1] if selected else [0]

      #USER_INPUT[4] = process_input(SH)
      if USER_INPUT[2] == 'Yes':
          USER_INPUT[2] = 1
      elif USER_INPUT[2] == 'No':
          USER_INPUT[2] = 0
      new_data = pd.DataFrame({'Age' : USER_INPUT[0], 'RTPCR test(CT value)' : USER_INPUT[1], 'Comorbidity' : USER_INPUT[2]}, index = [0])
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
    <h3 style="color: ##00FF00;">Diagonostic Recomendation</h3>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write('Input Data Array')
st.write(new_table_df)
bayes = load(open('content/bayes.pkl', 'rb'))
logistic = load(open('content/logistic.pkl', 'rb'))
random_tree =load(open('content/random_tree.pkl', 'rb'))
svm_linear = load(open('content/svm_linear.pkl', 'rb'))
svm_rbf = load(open('content/svm_rbf.pkl', 'rb'))
svm_sigmoid = load(open('content/svm_sigmoid.pkl', 'rb'))
tree = load(open('content/tree.pkl', 'rb'))
# Dropdown menu for model selection
selected_model = st.selectbox('Select ML Model', ['Naive Bayes Algorithm', 'Logistic Regression Algorithm', 'Decision Tree Algorithm', 'Random Forest Algorithm', 'SVM (Linear) Algorithm', 'SVM (RBF) Algorithm', 'SVM(Sigmoid) Algorithm'])
prediction=0
# Perform predictions based on the selected model

CSS = """
    <style>
        .header_pred{
                text-align: center;
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

if st.button('Make Prediction'):
    if selected_model == 'Naive Bayes Algorithm':
        prediction = bayes.predict(combined_df)
        st.write("Predicted Result:")
        #st.write(f"Fraction Value: {prediction*100}")
        if prediction == 1:
            st.markdown(CSS, unsafe_allow_html=True)
            st.markdown(HEAD_YES, unsafe_allow_html=True)
            st.cache_data.clear()
        else:
            st.cache_data.clear()
            st.markdown(CSS, unsafe_allow_html=True)
            st.markdown(HEAD_NO, unsafe_allow_html=True)
            st.cache_data.clear()
    elif selected_model == 'Logistic Regression Algorithm':
        prediction = logistic.predict(combined_df)
        st.write("Predicted Result:")
        #st.write(f"Fraction Value: {prediction*100}")
        if prediction == 1:
            st.markdown(CSS, unsafe_allow_html=True)
            st.markdown(HEAD_YES, unsafe_allow_html=True)
            st.cache_data.clear()
        else:
            st.cache_data.clear()
            st.markdown(CSS, unsafe_allow_html=True)
            st.markdown(HEAD_NO, unsafe_allow_html=True)
            st.cache_data.clear()
    elif selected_model == 'Decision Tree Algorithm':
        prediction = tree.predict(combined_df)
        st.write("Predicted Result:")
        #st.write(f"Fraction Value: {prediction*100}")
        if prediction == 1:
            st.markdown(CSS, unsafe_allow_html=True)
            st.markdown(HEAD_YES, unsafe_allow_html=True)
            st.cache_data.clear()
        else:
            st.cache_data.clear()
            st.markdown(CSS, unsafe_allow_html=True)
            st.markdown(HEAD_NO, unsafe_allow_html=True)
            st.cache_data.clear()
    elif selected_model == 'Random Forest Algorithm':
        prediction = random_tree.predict(combined_df)
        st.write("Predicted Result:")
        #st.write(f"Fraction Value: {prediction*100}")
        if prediction == 1:
            st.markdown(CSS, unsafe_allow_html=True)
            st.markdown(HEAD_YES, unsafe_allow_html=True)
            st.cache_data.clear()
        else:
            st.cache_data.clear()
            st.markdown(CSS, unsafe_allow_html=True)
            st.markdown(HEAD_NO, unsafe_allow_html=True)
            st.cache_data.clear()
    elif selected_model == 'SVM (Linear) Algorithm':
        prediction = svm_linear.predict(combined_df)
        st.write("Predicted Result:")
        #st.write(f"Fraction Value: {prediction*100}")
        if prediction == 1:
            st.markdown(CSS, unsafe_allow_html=True)
            st.markdown(HEAD_YES, unsafe_allow_html=True)
            st.cache_data.clear()
        else:
            st.cache_data.clear()
            st.markdown(CSS, unsafe_allow_html=True)
            st.markdown(HEAD_NO, unsafe_allow_html=True)
            st.cache_data.clear()
    elif selected_model == 'SVM (RBF) Algorithm':
        prediction = svm_rbf.predict(combined_df)
        st.write("Predicted Result:")
        #st.write(f"Fraction Value: {prediction*100}")
        if prediction == 1:
            st.markdown(CSS, unsafe_allow_html=True)
            st.markdown(HEAD_YES, unsafe_allow_html=True)
            st.cache_data.clear()
        else:
            st.cache_data.clear()
            st.markdown(CSS, unsafe_allow_html=True)
            st.markdown(HEAD_NO, unsafe_allow_html=True)
            st.cache_data.clear()
    elif selected_model == 'SVM (Sigmoid) Algorithm':
        prediction = svm_sigmoid.predict(combined_df)
        st.write("Predicted Result:")
        #st.write(f"Fraction Value: {prediction*100}")
        if prediction == 1:
            st.markdown(CSS, unsafe_allow_html=True)
            st.markdown(HEAD_YES, unsafe_allow_html=True)
            st.cache_data.clear()
        else:
            st.cache_data.clear()
            st.markdown(CSS, unsafe_allow_html=True)
            st.markdown(HEAD_NO, unsafe_allow_html=True)
            st.cache_data.clear()
