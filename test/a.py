
from langchain import hub
from langchain_community.chat_models import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain import PromptTemplate
# Prompt
#Improve This
prompt = PromptTemplate(
    template="""You are an expert doctor who is trained to give medical advice from the facts  \n
    Here are the facts:

    {context}

    Here is the query: {question}
     If you don't know the answer, just say that you don't know. Just Give the Answer in plain text""",
    input_variables=["question", "context"],
)
# LLM
# Post-processing
llm = Ollama(model="dhruvsingh959/ddx-llama")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = prompt | llm 

# Run
question = "I have pain in my ear"
generation = rag_chain.invoke({"context": "", "question": question})
print((generation))