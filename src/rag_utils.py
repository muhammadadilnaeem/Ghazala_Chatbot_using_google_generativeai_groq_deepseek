import os
from time import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_core.messages import AIMessage,HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv

load_dotenv()


gemini_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")



# set maximum documents limit for user
DB_DOCS_LIMIT = 10

# initialize chroma vector data base
def initialize_vectot_db(docs):
    vector_db = Chroma.from_documents(
            documents=docs,
            # embedding=GoogleGenerativeAIEmbeddings(api_key=gemini_api_key,model="models/embedding-001"),
            embedding=GoogleGenerativeAIEmbeddings(api_key=gemini_api_key,model="models/embedding-001"),
            collection_name=f"{str(time()).replace('.', '')[:14]}_" + st.session_state['session_id'],
        )

    # We need to manage the number of collections that we have in memory, we will keep the last 20
    chroma_client = vector_db._client
    collection_names = sorted([collection.name for collection in chroma_client.list_collections()])
    print("Number of collections:", len(collection_names))
    while len(collection_names) > 20:
        chroma_client.delete_collection(collection_names[0])
        collection_names.pop(0)
    return vector_db


# load url content
def load_url_to_db():
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        docs = []
        if url not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) < 10:
                
                try:
                    loader = WebBaseLoader(url)
                    docs.extend(loader.load())
                    st.session_state.rag_sources.append(url)
                
                except Exception as e:
                    st.error(f"Error Wwhile Loading Documents From {url}:{e}")
                
                if docs:
                    _split_and_load_docs(docs)
                    st.toast(f"Document From URL *{url}* Loaded Successfully", icon="✅")
                else:
                    st.error("Maximum Number Of Documents Reached 😨")


# split and load document chunks in vectordb
def _split_and_load_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 5000,
        chunk_overlap = 1000,
    )
    document_chunks = text_splitter.split_documents(docs)
    
    if "vector_db" not in st.session_state:
        st.session_state.vector_db =  initialize_vectot_db(docs)
    else:
        st.session_state.vector_db.add_documents(document_chunks)

# document indexing
def load_docs_to_db():
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = []
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    os.makedirs("source_files",exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    with open(file_path,"wb") as file:
                        file.write(doc_file.read())
                    try:
                        if doc_file.type == "application/pdf":
                            loader = PyPDFLoader(file_path)
                        elif doc_file.name.endswith(".docx"):
                            loader = Docx2txtLoader(file_path)
                        elif doc_file.type in ["text/plain","text/markdown"]:
                            loader = TextLoader(file_path)
                        else:
                            st.warning(f"We Are Sorry! Document type {doc_file.type} Not Supported.")
                            continue
                        docs.extend(loader.load())  
                        st.session_state.rag_sources.append(doc_file.name)
                    except Exception as e:
                           st.toast(f"Error Loading Document {doc_file.name}: {e}",icon="🙏")
                           print(f"Error Loading Document {doc_file.name}:{e}")

                    finally:
                        os.remove(file_path)
                else:
                    st.error(f"Maximum Number of Documents Reached {DB_DOCS_LIMIT}.")
        if docs:
            _split_and_load_docs(docs)
            st.toast(f"Document *{str([doc_file.name for doc_file in st.session_state.rag_docs])[1:-1]}* Loaded Successfully.", icon="✅")
            st.balloons()

def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get inforamtion relevant to the conversation, focusing on the most recent messages."),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

# # function to get conversation through rag
def get_conversational_rag_chain(llm):
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """You are a helpful assistant. Your name is Ghazala.You will have to answer to user's queries.
        You will have some context to help with your answers, but now always would be completely related or helpful.
        You can also use your knowledge to assist answering the user's queries.\n
        {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


# function to stream response of llm
def stream_llm_response(llm_stream, messages):
    response_message = ""
    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk
    st.session_state.messages.append({"role":"assistant","content": response_message})


# function to stream llm rag response
def stream_rag_llm_response(llm_stream, messages):
    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    response_message = "*(RAG Response)*\n"
    for chunk in conversation_rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
        response_message += chunk
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})

