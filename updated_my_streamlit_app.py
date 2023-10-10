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
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
    }
        </style>
    """

    head = """
        <h1 class="header">
            <font color="#2d00f7">Covid</font>
            <font color="#6a00f4">-</font>
            <font color="#8900f2">19</font>
            <font color="#a100f2">Diagonosis</font>
        </h1>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown(head, unsafe_allow_html=True)
header()
st.write("Results of Research work funded by ICMR ")
#col1, col2, col3 = st.columns(3)

USER_INPUT = [0, 0, 0, 0, 0, 0, 0, 0, 0]
def process_input(input_value):
    result = input_value
    return result

# Age Column.
# with col1:
#     AGE = st.number_input("Age", value=0)
#     USER_INPUT[0] = process_input(AGE)
AGE = st.number_input("Age", format="%.f")
USER_INPUT[0] = process_input(AGE)
# Bmi Column.
# with col1:
    # E_gene = st.number_input('CT value E gene', step=1.,format="%.f")
    # USER_INPUT[1] = process_input(E_gene)
E_gene = st.number_input('CT value E gene', step=1.,format="%.f")
USER_INPUT[1] = process_input(E_gene)
# HbA1c Column.
# with col1:
    # pre_medical = st.selectbox('Premedical Condition', ('Yes', 'No'))
    # USER_INPUT[2] = process_input(pre_medical)
pre_medical = st.selectbox('Premedical Condition', ('Yes', 'No'))
USER_INPUT[2] = process_input(pre_medical)
# Gender Column.
# with col1:
    # gender = st.selectbox('Gender', ('Male', 'Female'))
    # USER_INPUT[3] = process_input(gender)
gender = st.selectbox('Gender', ('Male', 'Female'))
USER_INPUT[3] = process_input(gender)

symptoms = ['fever','cough','breathlessness','body_ache','vomiting','sore_throat','diarrhoea','sputum','nausea','nasal_discharge','loss_of_taste','loss_of_smell','abdominal_pain','chest_pain','haemoptsis','head_ache','body_pain','weak_ness',
 'cold']

# Create a DataFrame to store symptom values
symptom_df = pd.DataFrame(columns=symptoms)
st.write("Symptoms Set")
# Create checkboxes for each symptom and update the DataFrame
for symptom in symptoms:
    selected = st.checkbox(symptom)
    symptom_df[symptom] = [1] if selected else [0]

#USER_INPUT[4] = process_input(SH)
if USER_INPUT[2] == 'Yes':
    USER_INPUT[2] = 1
elif USER_INPUT[2] == 'No':
    USER_INPUT[2] = 0
new_data = pd.DataFrame({'Age' : USER_INPUT[0], 'E_gene' : USER_INPUT[1], 'Pre_medical' : USER_INPUT[2]}, index = [0])
# Concatenate the two DataFrames vertically
combined_df = pd.concat([new_data, symptom_df], axis=1, ignore_index=True)
st.write(combined_df)
# combined_df.columns = list(new_data.columns) + list(symptom_df.columns)

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
# ####################################################### Button For Prediction #####################################################

# submit_button = st.button("Submit")
# if submit_button:
#     st.write("Combined DataFrame:")
#     st.write(combined_df)
    
bayes = pickle.load(open('/content/bayes.pkl', 'rb'))
logistic = pickle.load(open('/content/logistic.pkl', 'rb'))
random_tree = pickle.load(open('/content/random_tree.pkl', 'rb'))
svm_linear = pickle.load(open('/content/svm_linear.pkl', 'rb'))
svm_rbf = pickle.load(open('/content/svm_rbf.pkl', 'rb'))
svm_sigmoid = pickle.load(open('./content/svm_sigmoid.pkl', 'rb'))
tree = pickle.load(open('/content/tree.pkl', 'rb'))
# Dropdown menu for model selection
selected_model = st.selectbox('Select a Model', ['Naive Bayes', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM (Linear)', 'SVM (RBF)', 'SVM(Sigmoid)'])
prediction=0
# Perform predictions based on the selected model

if st.button('Make Predictions'):
    if selected_model == 'Naive Bayes':
        prediction = bayes.predict(combined_df)
        st.write("Predicted Results:")
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
        if prediction == 1:
            st.markdown(CSS, unsafe_allow_html=True)
            st.markdown(HEAD_YES, unsafe_allow_html=True)
            st.cache_data.clear()
        else:
            st.cache_data.clear()
            st.markdown(CSS, unsafe_allow_html=True)
            st.markdown(HEAD_NO, unsafe_allow_html=True)
            st.cache_data.clear()
# Perform predictions based on the Naive Bayes model
