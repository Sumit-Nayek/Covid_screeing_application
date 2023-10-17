import streamlit as st
import pickle
import pandas as pd
from pickle import load

def header():
    custom_css = """
        <style>
             .header{
                color: #fff;
                text-align: left;
                font-size: 66px;
                font-weight: bold;

    }
        </style>
    """

    head = """
        <h3 class="header">
            <font color="#2d00f7">Web-enabled Diagonosis for COVID-19</font>
            #<font color="#2d00f7">Diagonosis</font>
            #<font color="#2d00f7">for</font>
            #<font color="#2d00f7">COVID</font>
            #<font color="#2d00f7">-</font>
            #<font color="#2d00f7">19</font>
        </h3>
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
          <h4 style="color: ##00FF00;">Demographic Information</h4>

          </div>
          """,
          unsafe_allow_html=True,
      )
      patient_name =st.text_input("Patient's Name",f'<b>Patient\'s Name</b>')
      AGE = st.number_input(f'<b>Patient\'s Age</b>', format="%.f")
      USER_INPUT[0] = process_input(AGE)
      gender = st.selectbox(f"<b>Patient's Gender</b>", ('Male', 'Female'))
      USER_INPUT[3] = process_input(gender)
      state=st.text_input(f'<b>State</b>')
      country=st.text_input(f'<b>Country</b>')
      exposed=st.text_input('Exposed to covid infected zone',f'<b>Exposed to covid infected zone</b>')
      no_of_infected_person=st.text_input('Number of effected person in familiy',f'<b>Number of effected person in familiy</b>')
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
      symptoms = ['Fever', 'Cough', 'Breathlessness', 'Body ache', 'Vomiting', 'Sore throat',
                  'Diarrhoea', 'Sputum', 'Nausea', 'Nasal discharge', 'Loss of taste', 'Loss of smell',
                  'Abdominal pain', 'Chest pain', 'Haemoptsis', 'Head ache', 'Body pain', 'Cold']

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
      pre_medical = st.selectbox(f'<b>Comorbidity ?</b>', ('Yes', 'No'))
      USER_INPUT[2] = process_input(pre_medical)
      E_gene = st.number_input(f'<b>RTPCR test (CT value)</b>', step=1.,format="%.f")
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
      new_data = pd.DataFrame({'Age' : USER_INPUT[0], 'E_gene' : USER_INPUT[1], 'Pre_medical' : USER_INPUT[2]}, index = [0])
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
st.write('Input Array ')
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

if st.button('Make Predictions'):
    if selected_model == 'Naive Bayes':
        prediction = bayes.predict(combined_df)
        st.write("Predicted Results:")
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
    elif selected_model == 'Logistic Regression':
        prediction = logistic.predict(combined_df)
        st.write("Predicted Results:")
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
    elif selected_model == 'Decision Tree':
        prediction = tree.predict(combined_df)
        st.write("Predicted Results:")
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
    elif selected_model == 'Random Forest':
        prediction = random_tree.predict(combined_df)
        st.write("Predicted Results:")
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
    elif selected_model == 'SVM (Linear)':
        prediction = svm_linear.predict(combined_df)
        st.write("Predicted Results:")
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
    elif selected_model == 'SVM (RBF)':
        prediction = svm_rbf.predict(combined_df)
        st.write("Predicted Results:")
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
    elif selected_model == 'SVM (Sigmoid)':
        prediction = svm_sigmoid.predict(combined_df)
        st.write("Predicted Results:")
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
