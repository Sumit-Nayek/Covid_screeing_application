import streamlit as st
import pandas as pd
import base64
# from pickle import load
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Arc
import pgeocode
import requests
# Function to add a background image
  # OpenRouter API details
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

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
page = st.sidebar.selectbox("Go to", ["Diagonostic recomendation", "Descriptive Analysis","AI Assistant"])

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
if page == "Diagonostic recomendation":
    add_bg_from_local("content/new_test1.jpg")  # Background for Risk Assessment page
        
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
    CSS = """
        <style>
            .header_pred{
                    text-align: center;
                    font-size: 40px;
                    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
                }
        </style>
    """
    
    HEAD_YES = """
            <h4 class="header_pred" style="color:#affc42"> You Have Covid-19 </h6>
    """
    
    HEAD_NO = """
        <h4 class="header_pred" style="color:#affc42"> You Don't Have Covid-19 </h6>
    """
    ########################################## (New added part from sumit sir code ############
    def model_loader(model, new_data):
        scl = pkl.load(open('./models/Scaler.pkl', 'rb'))
        scaler = scl["stdscaler"]
        max_ct = scl["max_ct"]
        columnsN = new_data.columns
        new_data_std = scaler.transform(new_data) 
            # st.dataframe(new_data_std, hide_index= True)
        new_data =  pd.DataFrame(new_data_std,columns=columnsN) 
        # st.write("New data scaled")# Apply scaling on the test data
    
        # st.write(f'Using Model: {model}')
        try:
            match model:
                case 'Naive Bayesian':
                    load_model = pkl.load(open(f'./models/NB{max_ct}.pkl', 'rb'))
                case 'Decesion Tree':
                    load_model = pkl.load(open(f'./models/DT{max_ct}.pkl', 'rb'))
                case 'Random Forest':
                    load_model = pkl.load(open(f'./models/RF{max_ct}.pkl', 'rb'))
                case 'SVM (Linear)':
                    load_model= pkl.load(open(f'./models/SVM_linear{max_ct}.pkl', 'rb'))
                case 'SVM (RBF)':
                    load_model= pkl.load(open(f'./models/SVM_rbf{max_ct}.pkl', 'rb'))
                case 'SVM (Polynomial)':
                    load_model= pkl.load(open(f'./models/SVM_poly{max_ct}.pkl', 'rb'))
                case 'SVM (Sigmoidal)':
                    load_model= pkl.load(open(f'./models/SVM_sigmoid{max_ct}.pkl', 'rb'))
            
            result = load_model.predict(new_data)
            # st.write(result)
            # result = predict_results(st.session_state.load_model, new_data)
            diagonosis=None
            if result[0] == 1:
                st.markdown(CSS, unsafe_allow_html=True)
                diagonosis='Covid-19 Positive' 
                st.markdown(HEAD_YES, unsafe_allow_html=True)
                st.cache_data.clear()
                # st.subheader(f'You have Covid-19')
            else:
                st.markdown(CSS, unsafe_allow_html=True)
                diagonosis='Covid-19 Negative'
                st.markdown(HEAD_NO, unsafe_allow_html=True)
                st.cache_data.clear()
                # st.subheader(f'You don\'t have Covid-19')
            
        except FileNotFoundError:
                    st.error('Model not found. Please make sure the model file exists.')
        return diagonosis
   

    load_model = None
    features = pkl.load(open(f'./models/Features.pkl', 'rb'))
    keys = [features[x] for x in features.keys()]
    new_data = pd.DataFrame(columns=keys)
    # Load Indian postal code data

    with st.form('prsnlInfo', clear_on_submit=False):
        # st.header("Data Collection")
        c_00, c_01 = st.columns(2, gap="medium", vertical_alignment="top")
        with c_00:
            st.subheader('Personal Information')
            name = st.text_input('Name: ', placeholder='Enter Your Name')#, on_change = check_blank, args=(value,) )
            age = st.number_input("Age", step=1.,format="%.f")
            new_data.loc[0,'age'] = age
            sex = st.selectbox('Sex: ', options= ["Male", "Female"],)
            state = st.text_input('State: ', placeholder='Enter the State you are from')#, on_change=check_blank, args = (value,))
            # country_list = list(pycountry.countries)
            # cn_list = [c.name for c in country_list]
            # Taking Pincode as numeric input
            pincode = st.text_input("Enter Pincode", max_chars=6, placeholder="e.g., 110001")
            nomi = pgeocode.Nominatim('IN')
            location_info = nomi.query_postal_code(pincode)
            
            expo_infec = st.selectbox('Exposed to infected zone: ', options=["No", "Yes"])
            # st.write(location_info)
            # eff_mem = st.number_input('CT value E gene', min_value=0, max_value=10,step=1, format="%d")
        # next_sec = st.form_submit_button('Next Section')
        with c_01:
            st.subheader('Clinical Information')
            c_0, c_1 = st.columns(2, gap="medium", vertical_alignment="top")
            i = 0
            for feature, key in features.items():
                if feature not in ('Age','Comorbidity','RTPCR Test(CT VALUE)'): # 'Comorbidity', 
                    match i%2:
                        case 0:
                            with c_0:
                                new_data.loc[0,key] = 1 if st.checkbox(feature, key=key) else 0
                        case 1:
                            with c_1:
                                new_data.loc[0,key] = 1 if st.checkbox(feature, key=key) else 0
                    i += 1

                else:
                    if feature == 'Comorbidity':
                        new_data.loc[0,key] = 1 if st.selectbox(feature, options=["No", "Yes"]) == 'Yes' else 0
                                    
                    if feature == 'RTPCR Test(CT VALUE)':
                        new_data.loc[0,key] = st.number_input('CT value E gene', min_value=0, max_value=50,step=1, format="%d")
# if st.button("Save Data"):
        st.session_state.shared_data = new_data  # Store data globally
        # st.success("Data saved! Go to Risk Assesment Page to access it.")           
        
        st.header('Model Selection')
        modeli = st.selectbox('Select Model: ', options=['Naive Bayesian', 'Decesion Tree', 'Random Forest',
                            'SVM (Linear)', 'SVM (RBF)', 'SVM (Polynomial)', 'SVM (Sigmoidal)'],)
            # kernel = st.selectbox('Select Kernel: ', options=['Linear', 'RBF', 'Polynomial', 'Sigmoidal'],)
        btn_lm = st.form_submit_button('Predict')#, on_click=model_loader,args=(modeli, pd.DataFrame.from_dict(new_data)))
    if btn_lm:
        st.write("New data raw")
        st.dataframe(new_data)
        model_loader(modeli, new_data)  # Call model_loader function with selected model and new_data
    if st.button("Assess Risk"):
            # risk_score = calculate_risk_score(symptom_values, pre_medical1)
        symptom_values = new_data.iloc[0, 3:22]  # Extract all symptom columns
            # st.write(symptom_values)
        pre_medical = new_data.iloc[0, 1]  # Extract the last column (Pre-Medical Condition)
            # st.write(pre_medical)
        
            # Calculate risk score
        risk_score = sum(symptom_values.values)  # Sum of selected symptoms
        if pre_medical == 1:
            risk_score += 1  # Add 1 if pre-existing medical condition exists  
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

    prompt = f"""
        You are a resume parsing tool. Given the following resume text, extract only information that mentioned below but don't include any personal information and return them in a well-structured JSON format.
        Make sure all keys needs to be lowercase.
        The resume text:
        {resume_text}
        Extract and include the following:
        - Skills (include all skills at skill section)
        - Education (Only degree and major, and no university)
        - Experience (role and details at experience section)
        Please don't include any extra sentence or disclaimer"
    """
    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-3.2-3b-instruct:free",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for parsing resumes."},
                {"role": "user", "content": prompt},
            ]
        )

        if completion and completion.choices and completion.choices[0].message:
            return completion.choices[0].message.content
        else:
            raise ValueError("Invalid response structure from OpenAI API.")

    except Exception as e:
        return f"An error occurred: {str(e)}"
#####################

        
elif page == "Descriptive Analysis":
  # Adding a graph image (JPG format)
    image_path = "content/Risk_stratification_bar_diagram.jpg"  # Path to your JPG file
    st.image(image_path, caption="Multiple bar diagram of covid positive patients with comorbidity and symptoms of COVID-19 over different age groups and different risk labels (infectious nature)",  use_container_width=True)
    
    # Additional UI elements (optional)
    st.write("It was observed that that the bar length is high in positive patients who have comorbidity and symptoms all together. The middle age group shows symptoms with comorbidity; patients are mostly positive compared to other groups.")
    image_path = "content/Density_plot.jpg"  # Path to your JPG file
    st.image(image_path, caption="Density plot shows the distribution of age variable over symptomatic and comorbid covid positive patients",  use_container_width=True)
    
    # Additional UI elements (optional)
    st.write("From above plots one can see that for positive patients of comorbidity group the age is denser in the range 45-65 which means that the middle-aged peoples with comorbidity have more probability for coming positive in the RT-PCR test compared to younger peoples")
    image_path = "content/First_heatmap.jpg"  # Path to your JPG file
    st.image(image_path, caption="Heatmap of covid effected area in India",  use_container_width=True)
    
    # Additional UI elements (optional)
    st.write("Here the colour shade represent the magnitude of the covid impacted areas. Darker shades represnt the hotspot zones and lighter shades represent less impacted areas")
    image_path = "content/Heat_map_one.jpg"  # Path to your JPG file
    st.image(image_path, caption="Heatmap of covid effected area in India with satelite view",  use_container_width=True)
    
    # Additional UI elements (optional)
    st.write("Here the sizes of the buble represent the total number of covid effected people. Larger bubles represnt the most effected region and small bubles represent less effected regions")
    image_path = "content/Quaterly_trend_Covid.jpg"  # Path to your JPG file
    st.image(image_path, caption="Multiple bar diagram of covid positive male and female patients in different quarters of year 2020 and 2021.",  use_container_width=True)
    
    # Additional UI elements (optional)
    st.write("Here the percentage for each quarter is calculated with respect to the total number of positive cases throughout the covid period [Apr. 2020-Dec.2021]")



# Page 4: Primary Treatment
elif page == "AI Assistant":

    # Custom CSS for chat interface
    st.markdown(
        """
        <style>
        /* User message styling */
        .user-message {
            background-color: #0078D4;
            color: white;
            border-radius: 15px 15px 0 15px;
            padding: 10px;
            margin: 5px 0;
            max-width: 70%;
            margin-left: auto;
        }
    
        /* AI message styling */
        .ai-message {
            background-color: #F1F1F1;
            color: black;
            border-radius: 15px 15px 15px 0;
            padding: 10px;
            margin: 5px 0;
            max-width: 70%;
            margin-right: auto;
        }
    
        /* Chat container styling */
        .chat-container {
            display: flex;
            flex-direction: column;
            padding: 10px;
        }
    
        /* Streamlit chat input styling */
        .stChatInput {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: white;
            padding: 10px;
            box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    

    st.title("🤖 AI For Your Medical Assistance")
    
    # Check if DataFrame exists in session state
    if "shared_data" in st.session_state:
        df = st.session_state.shared_data  # Retrieve stored DataFrame
        # summary = df.describe().to_string()  # Generate a summary
        # st.write("📊 **Initial Data Analysis**")
        # st.dataframe(df.head())  # Display first few rows
    else:
        st.warning("No data found! Please upload it on the data page.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-container"><div class="user-message">{message["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-container"><div class="ai-message">{message["content"]}</div></div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask something about the data or medical guidance..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f'<div class="chat-container"><div class="user-message">{prompt}</div></div>', unsafe_allow_html=True)
    
        # Prepare the request payload for OpenRouter
        system_message = "You are a medical AI assistant. Use the provided patient data to give insights and guidelines. Also help them to retrive information from various sources"
        
        # if "shared_data" in st.session_state:
        #     system_message += f"\n\nHere is a summary of the patient data:\n{summary}"
        
        payload = {
            "model": "openai/gpt-3.5-turbo",
            "messages": st.session_state.messages + [{"role": "system", "content": system_message}],
        }
    
        # Send request to OpenRouter
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
    
        try:
            response = requests.post(OPENROUTER_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            ai_response = response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
            ai_response = "Sorry, I couldn't process your request. Please try again."
    
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        st.markdown(f'<div class="chat-container"><div class="ai-message">{ai_response}</div></div>', unsafe_allow_html=True)
    # # add_bg_from_local("content/primary_treatment_bg.jpg")  # Background for Primary Treatment page
    # header("Primary Treatment Instructions")

    # st.markdown(
    #     """
    #     <h3 style="color: #2d00f7;">General Guidelines for COVID-19 Management:</h3>
    #     <ul>
    #         <li>Isolate yourself to prevent the spread of infection.</li>
    #         <li>Stay hydrated and maintain a balanced diet.</li>
    #         <li>Monitor your symptoms regularly.</li>
    #         <li>Take over-the-counter medications for fever or pain as advised by your doctor.</li>
    #     </ul>
    #     <h3 style="color: #e500a4;">When to Seek Emergency Care:</h3>
    #     <ul>
    #         <li>Difficulty breathing or shortness of breath.</li>
    #         <li>Persistent chest pain or pressure.</li>                             
    #         <li>Confusion or inability to stay awake.</li>
    #         <li>Bluish lips or face.</li>
    #     </ul>
    #     """,
    #     unsafe_allow_html=True,
    # )
