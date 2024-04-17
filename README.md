# Health.AI

Health.AI is a personal healthcare assistant which is equipped with multimodal RAG (Retrieval Augmented Generation), capable of handling complex medical queries and asking follow-up questions to arrive at a differential diagnosis for the patient along with relevant suggestions. 

The chatbot is powered by LLaVA and [ddx-llama](https://ollama.com/dhruvsingh959/ddx-llama), a version of the open-source LLama 2, which we fine-tuned on our own [dataset](https://huggingface.co/datasets/satyam-03/opdx-dataset) consisting of 10k samples of medical conversations between a doctor and a patient that we generated using the publicly available [DDXPlus](https://github.com/mila-iqia/ddxplus) dataset of synthetic patient symptoms, antecedents and their respective differential diagnoses. 

![Screenshot from 2024-04-16 11-50-39](https://github.com/DhruvSinghiitmandi/Health.ai/assets/126661857/ce01b2d1-a275-4f7a-a308-cb0e92d5a499)

Our user interface integrates **OAuth** and the **Google Fit REST API** to provide real-time data (fetched from phones and smart wearables if available) on the user dashboard. This enables the user to monitor their personal health and fitness metrics and make queries to our chatbot accordingly.

The application currently supports multimodality in the form of **PDF** documents or **PNG** images, which the user can upload from their device to provide any sort of context for the queries.

Our early testing and analysis shows that our LLM outperforms publicly accessible models like GPT-3.5 and Gemini for medical question-answering.

## Multi-Modal RAG Methodology

![RAG](https://github.com/DhruvSinghiitmandi/Health.ai/assets/126661857/55bc923a-d534-4e37-9be7-ce2101d49593)

## Installation 
(Tested on Ubuntu 22.04)

#### Install Dependencies 
```
sudo apt-get install poppler-utils tesseract-ocr
curl -fsSL https://ollama.com/install.sh | sh
ollama pull dhruvsingh959/ddx-llama
```

#### Run

Paste the following commands in your terminal:
```
git clone https://github.com/dhruvsinghiitmandi/health.ai.git
cd health.ai
pip install -r requirements.txt
langchain serve
```

Open a new terminal and paste the following commands:
```
cd stlit
streamlit run Account.py 
```
The application should be up and running in your browser.

###




 

