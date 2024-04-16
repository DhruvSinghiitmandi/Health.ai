import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError
import google.generativeai as genai
import time
import ollama
import json


# Load authorization token from file
try:
    with open("token.json", "r") as f:
        token_file = json.load(f)
        token = token_file["token"]
except FileNotFoundError:
    print("Error: Please Sign In first !")
    exit(1)

from query import fetch_fitness_data

fit_data = fetch_fitness_data(token)

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(page_title="Advisor", page_icon="🤖", layout="wide")
# st.set_page_config(layout="wide")

# Streamed response emulator
def response_generator(prompt):
    text = ollama.chat(
    model='healthai',
    messages=[{'role': 'user', 'content': prompt}],
    stream=False
    )
    print(text)
    for word in text['message']['content'].split():
        yield word + " "
        time.sleep(0.05)

# st.title("HealthAI")

# Left-aligned greeting using Markdown and HTML

def colored_markdown(text: str, color: str):
    return f'<p class="title" style="background-color: #fff; color: {color}; padding: 5px; line-height: 1">{text}</p>'

sidebar = st.sidebar

# Greeting with custom colors
st.markdown(colored_markdown(f"Hello, {fit_data['name']}", "#007bff"),
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

# st.markdown("""
# <div class="square-container">
#     <div class="square">Give details of diseases caused by abnormal haemoglobin <img class="square-image" src="https://i.imgur.com/2Ja3RCi.png"></div>
#     <div class="square">Suggest a disease caused by these symptoms <img class="square-image" src="https://i.imgur.com/2Ja3RCi.png"></div>
#     <div class="square">What is this image about<img class="square-image" src="https://i.imgur.com/2Ja3RCi.png"></div>
#     <div class="square">can you identify the place of injury in this image<img class="square-image" src="https://i.imgur.com/2Ja3RCi.png"></div>
# </div>
# """, unsafe_allow_html=True)

# st.markdown("""
# <div class="input-container">
#     <input class="input-text" type="text" placeholder="Type here...">
#     <div class="button-container">
#         <div class="button" onclick="addImage()">
#             <svg width="24" height="24" fill="#39A5A9" xmlns="http://www.w3.org/2000/svg">
#                 <path d="M18.75 4C19.1768 4 19.5994 4.08406 19.9937 4.24739C20.388 4.41072 20.7463 4.65011 21.0481 4.9519C21.3499 5.25369 21.5893 5.61197 21.7526 6.00628C21.9159 6.40059 22 6.8232 22 7.25V18.75C22 19.1768 21.9159 19.5994 21.7526 19.9937C21.5893 20.388 21.3499 20.7463 21.0481 21.0481C20.7463 21.3499 20.388 21.5893 19.9937 21.7526C19.5994 21.9159 19.1768 22 18.75 22H7.25C6.8232 22 6.40059 21.9159 6.00628 21.7526C5.61197 21.5893 5.25369 21.3499 4.9519 21.0481C4.65011 20.7463 4.41072 20.388 4.24739 19.9937C4.08406 19.5994 4 19.1768 4 18.75V12.502C4.474 12.7 4.977 12.842 5.5 12.924V18.75C5.50067 18.9587 5.535 19.1567 5.603 19.344L11.426 13.643C11.8252 13.2522 12.3554 13.0239 12.9136 13.0025C13.4718 12.981 14.018 13.168 14.446 13.527L14.574 13.643L20.396 19.345C20.464 19.1583 20.4987 18.96 20.5 18.75V7.25C20.5 6.78587 20.3156 6.34075 19.9874 6.01256C19.6592 5.68437 19.2141 5.5 18.75 5.5H12.924C12.844 4.98489 12.7023 4.48127 12.502 4H18.75ZM12.559 14.644L12.475 14.714L6.668 20.401C6.85067 20.4657 7.04467 20.4987 7.25 20.5H18.75C18.953 20.5 19.149 20.465 19.33 20.401L13.525 14.715C13.3984 14.5909 13.2318 14.5156 13.055 14.5026C12.8782 14.4896 12.7024 14.5397 12.559 14.644ZM16.252 7.5C16.8493 7.5 17.4221 7.73726 17.8444 8.1596C18.2667 8.58193 18.504 9.15473 18.504 9.752C18.504 10.3493 18.2667 10.9221 17.8444 11.3444C17.4221 11.7667 16.8493 12.004 16.252 12.004C15.6547 12.004 15.0819 11.7667 14.6596 11.3444C14.2373 10.9221 14 10.3493 14 9.752C14 9.15473 14.2373 8.58193 14.6596 8.1596C15.0819 7.73726 15.6547 7.5 16.252 7.5ZM6.5 1C7.22227 1 7.93747 1.14226 8.60476 1.41866C9.27205 1.69506 9.87837 2.10019 10.3891 2.61091C10.8998 3.12163 11.3049 3.72795 11.5813 4.39524C11.8577 5.06253 12 5.77773 12 6.5C12 7.22227 11.8577 7.93747 11.5813 8.60476C11.3049 9.27205 10.8998 9.87837 10.3891 10.3891C9.87837 10.8998 9.27205 11.3049 8.60476 11.5813C7.93747 11.8577 7.22227 12 6.5 12C5.04131 12 3.64236 11.4205 2.61091 10.3891C1.57946 9.35764 1 7.95869 1 6.5C1 5.04131 1.57946 3.64236 2.61091 2.61091C3.64236 1.57946 5.04131 1 6.5 1ZM16.252 9C16.1532 9 16.0555 9.01945 15.9642 9.05724C15.873 9.09503 15.7901 9.15043 15.7203 9.22026C15.6504 9.29009 15.595 9.37299 15.5572 9.46422C15.5195 9.55546 15.5 9.65325 15.5 9.752C15.5 9.85075 15.5195 9.94854 15.5572 10.0398C15.595 10.131 15.6504 10.2139 15.7203 10.2837C15.7901 10.3536 15.873 10.409 15.9642 10.4468C16.0555 10.4845 16.1532 10.504 16.252 10.504C16.4514 10.504 16.6427 10.4248 16.7837 10.2837C16.9248 10.1427 17.004 9.95144 17.004 9.752C17.004 9.55256 16.9248 9.36128 16.7837 9.22026C16.6427 9.07923 16.4514 9 16.252 9ZM6.5 3L6.41 3.007C6.3101 3.02525 6.21812 3.07349 6.14631 3.14531C6.07449 3.21712 6.02625 3.3091 6.008 3.409L6 3.5V6H3.498L3.408 6.008C3.3081 6.02625 3.21612 6.07449 3.14431 6.14631C3.07249 6.21812 3.02425 6.3101 3.006 6.41L2.998 6.5L3.006 6.59C3.02425 6.6899 3.07249 6.78188 3.14431 6.85369C3.21612 6.92551 3.3081 6.97375 3.408 6.992L3.498 7H6V9.503L6.008 9.593C6.02625 9.6929 6.07449 9.78488 6.14631 9.85669C6.21812 9.92851 6.3101 9.97675 6.41 9.995L6.5 10.004L6.59 9.995C6.6899 9.97675 6.78188 9.92851 6.85369 9.85669C6.92551 9.78488 6.97375 9.6929 6.992 9.593L7 9.503V7H9.505L9.595 6.992C9.6949 6.97375 9.78688 6.92551 9.85869 6.85369C9.93051 6.78188 9.97875 6.6899 9.997 6.59L10.005 6.5L9.997 6.41C9.97868 6.30996 9.93029 6.21788 9.85829 6.14605C9.78628 6.07423 9.69409 6.02607 9.594 6.008L9.504 6H7V3.5L6.992 3.41C6.97393 3.30991 6.92577 3.21772 6.85395 3.14571C6.78212 3.07371 6.69004 3.02532 6.59 3.007L6.5 3Z"/>
#             </svg>
#         </div>
#         <div class="button" onclick="addMic()">
#             <svg width="24" height="24" fill="#39A5A9" xmlns="http://www.w3.org/2000/svg">
#                 <path d="M12 14C11.1667 14 10.4583 13.7083 9.875 13.125C9.29167 12.5417 9 11.8333 9 11V5C9 4.16667 9.29167 3.45833 9.875 2.875C10.4583 2.29167 11.1667 2 12 2C12.8333 2 13.5417 2.29167 14.125 2.875C14.7083 3.45833 15 4.16667 15 5V11C15 11.8333 14.7083 12.5417 14.125 13.125C13.5417 13.7083 12.8333 14 12 14ZM11 21V17.925C9.26667 17.6917 7.83333 16.9167 6.7 15.6C5.56667 14.2833 5 12.75 5 11H7C7 12.3833 7.48767 13.5627 8.463 14.538C9.43833 15.5133 10.6173 16.0007 12 16C13.3827 15.9993 14.562 15.5117 15.538 14.537C16.514 13.5623 17.0013 12.3833 17 11H19C19 12.75 18.4333 14.2833 17.3 15.6C16.1667 16.9167 14.7333 17.6917 13 17.925V21H11Z" />
#             </svg>
#         </div>
#         <div class="button" onclick="addCustom()">
#             <svg width="24" height="24" fill="#39A5A9" xmlns="http://www.w3.org/2000/svg">
#                 <path d="M3 20V4L22 12L3 20ZM5 17L16.85 12L5 7V10.5L11 12L5 13.5V17Z" />
#             </svg>
#         </div>
#     </div>
# </div>
# """, unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# st.header("Ask Anything ! 🤖")
uploaded_file = st.file_uploader(
    "Upload a pdf or image file",
    type=["pdf", "jpg", "png", "jpeg"],
    help="Upload a file to ask related questions.",
)

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        st.write("PDF file uploaded successfully.")
    else:
        st.write("Image file uploaded successfully.")

st.sidebar.write("##")
st.sidebar.markdown("<h2 style='text-align: center;'>User Dashboard</h2>", unsafe_allow_html=True)
st.sidebar.write("##")

col1, col2 = st.sidebar.columns(2)
col1.metric(label="Distance Covered", value=int(fit_data['distance']), delta=-150)
col2.button("View Details", key="distance")

col1, col2 = st.sidebar.columns(2)
col1.metric(label="Step Count", value=fit_data['steps'], delta=123)
col2.button("View Details", key="steps")

col1, col2 = st.sidebar.columns(2)
col1.metric(label="Calories Burned", value=int(fit_data['calories']), delta=0.5, delta_color="inverse")
col2.button("View Details", key="calories")

col1, col2 = st.sidebar.columns(2)
col1.metric(label="Heart Rate", value=int(fit_data['heart_rate']), delta_color="off", delta=-2)
col2.button("View Details", key="heart_rate")


