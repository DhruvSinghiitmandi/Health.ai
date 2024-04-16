from fastapi import FastAPI , HTTPException
from fastapi.responses import RedirectResponse
from langserve import add_routes
from typing import Dict
app = FastAPI()
# -- coding: utf-8 --


import os
#os.environ['COHERE_API_KEY'] = <your-api-key>
os.environ['TAVILY_API_KEY'] = "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"


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
### from langchain_cohere import CohereEmbeddings
import uuid
import subprocess
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
subprocess.Popen("ollama pull medllama2:7b-q4_0", shell=True)
subprocess.Popen("ollama pull dhruvsingh959/ddx-llama", shell=True)
subprocess.Popen("ollama pull llava:7b-v1.5-q4_0", shell=True)

vectorstore = Chroma(
    collection_name="rag-chroma",

    embedding_function=HuggingFaceEmbeddings(),
)
from langchain_community.llms import Ollama
llava = Ollama(model="llava:7b-v1.5-q4_0")

# original retriever = vectorstore.as_retriever()
store = InMemoryStore()
id_key = "doc_id"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

import time
import urllib.request

def upload_pdf(url):
    url = "https://sgp.fas.org/crs/misc/IF10244.pdf"
    filename = f"uploaded_file.pdf"
    # urllib.request.urlretrieve(url, filename)
    path = "./"
    from typing import Any

    from pydantic import BaseModel
    from unstructured.partition.pdf import partition_pdf
    # Extract images, tables, and chunk text
    raw_pdf_elements = partition_pdf(
        filename=path + filename,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=path,
    )
    category_counts = {}

    for element in raw_pdf_elements:
        category = str(type(element))
        if category in category_counts:
            category_counts[category] += 1
        else:
            category_counts[category] = 1

    # Unique_categories will have unique elements
    # TableChunk if Table > max chars set above
    unique_categories = set(category_counts.keys())
    category_counts
    class Element(BaseModel):
        type: str
        text: Any

    # Categorize by type
    categorized_elements = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            categorized_elements.append(Element(type="table", text=str(element)))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            categorized_elements.append(Element(type="text", text=str(element)))

    # Tables
    table_elements = [e for e in categorized_elements if e.type == "table"]
    print(len(table_elements))

    # Text
    text_elements = [e for e in categorized_elements if e.type == "text"]
    print(len(text_elements))


    #!curl -fsSL https://ollama.com/install.sh | sh
    import subprocess
    subprocess.Popen("ollama serve", shell=True)
    
    #!sleep 10

    from langchain_community.llms import Ollama
    subprocess.Popen("ollama pull medllama2:7b-q4_0", shell=True)
    subprocess.Popen("ollama pull dhruvsingh959/ddx-llama", shell=True)
    subprocess.Popen("ollama pull llava:7b-v1.5-q4_0", shell=True)

                        
    llm = Ollama(model="dhruvsingh959/ddx-llama")
    from langchain_community.chat_models import ChatOllama
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.embeddings import HuggingFaceEmbeddings
    import os
    import sys
    from langchain.llms import Replicate
    from langchain.vectorstores import Pinecone
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.document_loaders import PyPDFLoader
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.chains import ConversationalRetrievalChain
    # Prompt
    from langchain_community.llms import Ollama
    llm = Ollama(model="dhruvsingh959/ddx-llama")
    prompt_text = """You are an assistant tasked with summarizing tables and text. \
    Give a concise summary of the table or text. Table or text chunk: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    summarize_chain = {"element": lambda x: x} | prompt | llm | StrOutputParser()

    texts = [i.text for i in text_elements if i.text != ""]
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    # Apply to tables
    tables = [i.text for i in table_elements]
    table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

    # Add texts
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(text_summaries)
    ]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, texts)))

    # Add tables
    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=s, metadata={id_key: table_ids[i]})
        for i, s in enumerate(table_summaries)
    ]
    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(table_ids, tables)))
    import glob

    import base64
    import subprocess
    subprocess.Popen("ollama serve", shell=True)
    #!sleep 10

    
    def image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return encoded_string.decode("utf-8")

    IMG_DIR = './'

    # Use glob to match file paths
    image_files = glob.glob(IMG_DIR + "*.jpg")
    print(image_files)
    cleaned_img_summary = []

    # Iterate over matched file paths
    for img in image_files:
        # Perform your operation here
        RES = llava(prompt="Provide a concise, factual summary of the image, capturing all the key visual elements and details you observe. Avoid speculative or imaginative descriptions not directly supported by the contents of the image. Focus on objectively describing what is present in the image without introducing any external information or ideas. Your summary should be grounded solely in the visual information provided." ,images=[str(image_to_base64(img))])
        cleaned_img_summary.append(RES)
    img_ids = [str(uuid.uuid4()) for _ in cleaned_img_summary]
    summary_img = [
        Document(page_content=s, metadata={id_key: img_ids[i]})
        for i, s in enumerate(cleaned_img_summary)
    ]
    print(cleaned_img_summary)
    retriever.vectorstore.add_documents(summary_img)
    retriever.docstore.mset(
        list(zip(img_ids, cleaned_img_summary))
    )
    return "200"
import base64
# abc = upload_pdf("url")
def image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return encoded_string.decode("utf-8")


def upload_single_img(filename):
    subprocess.Popen("ollama pull llava:7b-v1.5-q4_0", shell=True)

    cleaned_img_summary = []
    print("here")
    filename = 'uploaded_file.png'
# Iterate over matched file paths
    for img in [filename]:
        # Perform your operation here
        print("before LLAva")
        RES = llava(prompt="Provide a concise, factual summary of the image, capturing all the key visual elements and details you observe. Avoid speculative or imaginative descriptions not directly supported by the contents of the image. Focus on objectively describing what is present in the image without introducing any external information or ideas. Your summary should be grounded solely in the visual information provided." ,images=[str(image_to_base64('./'+img))])
        cleaned_img_summary.append(RES)
    print("out")
    img_ids = [str(uuid.uuid4()) for _ in cleaned_img_summary]
    summary_img = [
        Document(page_content=s, metadata={id_key: img_ids[i]})
        for i, s in enumerate(cleaned_img_summary)
    ]
    print(cleaned_img_summary)
    retriever.vectorstore.add_documents(summary_img)
    retriever.docstore.mset(
        list(zip(img_ids, cleaned_img_summary))
    )






@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# @app.post("/generate/")
# async def generate_response(data: Dict[str, str]):
#     try:
#         # Extract input from request body
#         return "HI"
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000"
    "http://localhost:8080",
]
from langchain_community.chat_models import ChatOllama
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from langchain_core.output_parsers import StrOutputParser

@app.post("/upload_img")
async def up_img(inp:  str)-> Dict[str,str]:
    try:
        upload_single_img(inp)
        return {"message":"200"}
    except:
        print(Exception)
        return {"message":"404"}

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

@app.post("/generate")

async def generate_output(inp: str) -> Dict[str, str]:
    s = "You Are my personal doctor. You have to Remember my symptoms antecendants past history and try to ask me follow up questions whenever I tell you that I am not well"

# Prompt template
    template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. {{ if .}}### Instruction: You are a my personal doctor. You have to Remember my symotoms antecendants past history and try to ask me follow up questions whenever I Tell you that I am not well . You also have the following context, which can include text and tables to answer my QUESTION {{ .System }}{{ end }} {{ if .Prompt }}{context}
Question: {question}{{ .Prompt }}{{ end }} ### Response:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Option 1: LLM
    model = ChatOllama(model="dhruvsingh959/ddx-llama")
    # Option 2: Multi-modal LLM
    # model = GPT4-V or LLaVA

    # RAG pipeline
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    txt = chain.invoke(
     inp
    )








    

    # Final generation
    return {"message":txt}

@app.post("/upload")
async def post_pdf(url: str) -> Dict[str,str]:
    x = "404"
    try:
        x = await upload_pdf(url)
        return {"message":x}
    except:
        return {"message":x}

if __name__ =="__main__":
    uvicorn.run(app,host="localhost",port=8000, reload=False)