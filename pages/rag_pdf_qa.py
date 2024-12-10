import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

def initialize_llm(apikey):
    # llm = ChatGroq(api_key=apikey, model="llama-3.1-70b-versatile")
    llm = ChatOpenAI(api_key=apikey, model="gpt-4o-mini")
    return llm


def load_file(upload_file):
    documents = []
    temp_file = f"./uploaded_file.pdf"
    with open(temp_file, 'wb') as file:
        file.write(upload_file.read())
    loader = PyPDFLoader(temp_file)
    docs = loader.load()
    documents.extend(docs)
    os.remove(temp_file)

    return documents


def create_vectorstore(documents, embeddings):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore


def setup_chain(llm, vector_store):
    retriever = vector_store.as_retriever()
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question in Japanese. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


st.title("RAG with PDF upload")
st.write("Upload PDF and ask about content")

with st.sidebar:
    api_key = st.text_input("Enter your Groq API key:", type="password")


if api_key:
    llm = initialize_llm(api_key)

    upload_file = st.file_uploader("Choose PDF file", type="pdf")
    if upload_file:
        print(type(upload_file))
        print(upload_file)
        documents = load_file(upload_file)
        embeddings = HuggingFaceEmbeddings()
        vector_store = create_vectorstore(documents, embeddings)

        rag_chain = setup_chain(llm, vector_store)

        user_input = st.text_input("Your question:")
        if user_input:
            response = rag_chain.invoke({"input": user_input})
            st.write(response["answer"])

else:
    st.warning("Please enter Groq API Key")
