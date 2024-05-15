import uvicorn ##ASGI
from fastapi import FastAPI

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import warnings
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from typing import Any
from langchain.agents import AgentExecutor
from langchain_core.tools import tool

from dotenv import load_dotenv

load_dotenv()
groq_api_key=os.getenv('GROQ_API_KEY')

app = FastAPI(
    title="RAG_api",
    version="1.0",
    description="For RAG of rules and regulations"
)

warnings.filterwarnings('ignore')

##Create checklist to choose the model to run
llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="mixtral-8x7b-32768")

prompt=ChatPromptTemplate.from_template(
"""
You are a Mineral Exploration assistant.You are given a context and an input query. Use the context to answer the query.
<context>
{context}
<context>
Questions:{input}

"""
)

@app.get("/Embed_Documents")
def vector_embedding():

    if "vectors" not in st.session_state:

        
        st.write("Embeddings defined")
        st.session_state.loader=PyPDFDirectoryLoader("./rules_regulations") ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.write(st.session_state.docs)
        st.write("PDF loaded")
        st.write(len(st.session_state.docs))
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=4000,chunk_overlap=500) ## Chunk Creation
        st.write("Chunks created")
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs) #splitting
        st.write("Splitting done")
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector embeddings
        st.session_state.vectors.save_local("Faiss_index")
        st.write("Embeddings created")
    else:
        st.write("Faiss index already exist")
        




embeddings=HuggingFaceEmbeddings()
vectors = FAISS.load_local("Faiss_index", embeddings, allow_dangerous_deserialization=True)

@app.post("/Get_response/{query}")
def Get_Rag_Response(query):
    if query:
        print("Got query:")
        print(query)
        document_chain=create_stuff_documents_chain(llm,prompt)
        retriever=vectors.as_retriever()
        print("Retrieved docs")
        retrieval_chain=create_retrieval_chain(retriever,document_chain)
        response=retrieval_chain.invoke({'input':query})
        return response
        
    else:
        print("Error: No query passed")

# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8000)

# PORT = os.getenv('PORT')
