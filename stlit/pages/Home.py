import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError
import google.generativeai as genai
import time
import ollama
import json
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
import streamlit.components.v1 as com
import subprocess

# Load authorization token from file
try:
    with open("token.json", "r") as f:
        token_file = json.load(f)
        token = token_file["token"]
except FileNotFoundError:
    print("Error: Please Sign In first !")
    st.write("Please Sign In to continue to Chat")
    exit(1)

from query import fetch_fitness_data
from query import fetch_fitness_data_old
from query import fetch_name
fit_data = fetch_fitness_data(token)
old_fit_data = fetch_fitness_data_old(token)
name = fetch_name(token)

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(page_title="Health.AI", page_icon="🩺", layout="wide")

# Streamed response emulator
import requests
def response_generator(prompt):
    response = requests.post(f'http://127.0.0.1:8000/generate?inp={prompt}')
    text = response.json()['message']
    print(text)
    for word in text.split():
        yield word + " "
        time.sleep(0.05)

def colored_markdown(text: str, color: str):
    return f'<p class="title" style="background-color: #fff; color: {color}; padding: 5px; line-height: 1">{text}</p>'

sidebar = st.sidebar
col1, col2 = st.columns([1, 3])

# Display Lottie animation in the first column
with col1:
    com.iframe("https://lottie.host/embed/5899ceed-3498-4546-8ebf-b25561f40002/Xnif8r8nZ4.json", height=400, width=950)

try:
    st.markdown(colored_markdown(f"{fit_data['name']}", "#007bff"),
                unsafe_allow_html=True)  # Blue color
    st.markdown(colored_markdown("How can I help you today ?", "#39A5A9"),
                unsafe_allow_html=True)  # Red color
except:

    st.markdown(colored_markdown(f"{name['name']}", "#007bff"),
                unsafe_allow_html=True)  # Blue color
    st.markdown(colored_markdown("How can I help you today ?", "#39A5A9"),
                unsafe_allow_html=True)  # Red color

st.markdown(
    """
    <style>
        .title {
            text-align: left; 
            font-size: 60px;
            margin-bottom: 0px;
            padding-bottom: 5px;
            display: inline-block;  
        }
        .subtitle {
            text-align: left;
            font-size: 24px;
            color: #333333; 
            margin-top: 5px; 
        }
        .square-container {
            display: flex;
            flex-wrap: wrap;
        }
        .square {
            width: 150px;
            height: 150px;
            background-color: #36A5A9;
            margin: 10px;
            margin-top: 30px;  
            margin-bottom: 50px;  
            color: #ffffff;
            padding: 10px;
            text-align: left;
            font-size: 14px;
            line-height: 1.5;
            border-radius: 16px;
            position: relative;  /* Enable relative positioning for image */
        }
        .square-image {
            position: absolute;  /* Make image absolute within square */
            bottom: 5px;  /* Position image at bottom */
            right: 5px;  /* Position image at right */
            width: 20px;
            height: 20px;
        }
        .input-container {
            display: flex;
            align-items: center;
            position: relative;
            margin-top: 20px;
        }
        .input-text {
            flex: 1;
            height: 40px;
            padding: 10px;
            font-size: 16px;
            border-radius: 12px;
            border: 1px solid #ccc;
            margin-right: 10px;
        }
        .button-container {
            display: flex;
            gap: 0px;
        }
        .button {
            
            height: 40px;
            width: 40px;
            margin: 0px;
            padding: 0px;
            display: flex;
            justify-content: center;
            align-items: center;
            # background-color: #39A5A9;
            color: #fff;
            border-radius: 8px;
            cursor: pointer;
        }
        .button svg {
            width: 24px;
            height: 24px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(f'<img src="https://i.imgur.com/ngr2HSn.png" width="200">',
                    unsafe_allow_html=True)  # Load image from Imgur with Bitly link

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})

# File uploader
uploaded_file = st.file_uploader(
    "Upload a pdf or image file",
    type=["pdf", "jpg", "png", "jpeg"],
    help="Upload a file to ask related questions."
)

import os
# Check if a file is uploaded
if uploaded_file is not None:
    print("File uploaded")  # Debugging statement
    
    # Check the type of the uploaded file
    if uploaded_file.type == "application/pdf":
        file_extension = "pdf"
    else:
        file_extension = uploaded_file.name.split(".")[-1]
        
    
    # Get the parent directory path
    parent_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    
    # Save the file in the parent directory
    with open(os.path.join(parent_directory, f"uploaded_file.{file_extension}"), "wb") as f:
        f.write(uploaded_file.getbuffer())
        print('Success')  # Debugging statement

    
    # Display a success message
    st.write(f"File uploaded successfully: {uploaded_file.name}")
    if uploaded_file.type == "application/pdf":
        pdf_response = requests.post(f'http://127.0.0.1:8000/upload?url=uploaded_file.pdf')
        subprocess.Popen("rm ./../uploaded_file*",shell=True)
    else:
        img_response = requests.post(f'http://127.0.0.1:8000/upload_img?inp=uploaded_file.png')
        subprocess.Popen("rm ./../uploaded_file*",shell=True)

st.sidebar.write("##")
st.sidebar.markdown("<h2 style='text-align: center;'>User Dashboard</h2>", unsafe_allow_html=True)
st.sidebar.write("##")

if fit_data != 1:
    col1, col2 = st.sidebar.columns(2)

    col1.metric(label="Distance Covered", value=int(fit_data['distance']), delta=int(fit_data['distance'])-int(old_fit_data['distance']))
    col2.button("View Details", key="distance")
    
    col1, col2 = st.sidebar.columns(2)
    col1.metric(label="Step Count", value=fit_data['steps'], delta=int(fit_data['steps'])-int(old_fit_data['steps']))
    col2.button("View Details", key="steps")

    col1, col2 = st.sidebar.columns(2)
    col1.metric(label="Calories Burned", value=int(fit_data['calories']), delta=int(fit_data['calories'])-int(old_fit_data['calories']))
    col2.button("View Details", key="calories")

    col1, col2 = st.sidebar.columns(2)
    col1.metric(label="Heart Rate", value=int(fit_data['heart_rate']), delta=int(fit_data['heart_rate'])-int(old_fit_data['heart_rate']))
    col2.button("View Details", key="heart_rate")


