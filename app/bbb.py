from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os

os.environ['OPENAI_API_KEY']="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"

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    decsription="A simple API Server"

)

def hi(prompt: str) -> str:
    """
    A simple function that takes a prompt as input and returns the same prompt.
    
    Args:
        prompt (str): The input prompt.
        
    Returns:
        str: The same prompt that was passed as input.
    """
    return prompt


def query(inp: str) -> str:
    """
    A simple function that takes a prompt as input and returns the same prompt.
    
    Args:
        prompt (str): The input prompt.
        
    Returns:
        str: The same prompt that was passed as input.
    """
    return inp


prompt=ChatPromptTemplate.from_template("provide me an essay about {topic}")

add_routes(
    app,
    prompt | query,
    path="/openai"
)
print(type(ChatOpenAI()))

model=ChatOpenAI()
prompt=ChatPromptTemplate.from_template("provide me an essay about {topic}")
prompt1=ChatPromptTemplate.from_template("provide me a poem about {topic}")

add_routes(
    app,
    prompt|model,
    path="/essay"

)

add_routes(
    app,
    prompt1|model,
    path="/poem"

)



if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)