import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv


load_dotenv()

# LangSmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.environ.get('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = os.environ.get('LANGCHAIN_TRACING_V2')
os.environ['LANGCHAIN_PROJECT'] = os.environ.get('LANGCHAIN_PROJECT')

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response to the user queries."),
        ("user", "Question:{question}")
    ]
)


def generate_response(question, api_key, model, temperature, max_tokens):
    openai.api_key = api_key
    llm = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer


with st.sidebar:
    st.title("Setting")
    apikey = st.text_input("Please Enter your OpenAI API Key", type="password")
    # Drop down to select OpenAI models
    model = st.selectbox("Select model", ['gpt-4o-mini', 'gpt-4o', 'gpt-4o-turbo'])

    # Adjust response parameters
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
    max_tokens = st.slider("Max tokens", min_value=100, max_value=500, value=250)


st.title("QA Chatbot with OpenAI")
st.write("Go ahead and ask question")
user_input = st.text_input("Your question: ")


if user_input and apikey:
    response = generate_response(user_input, apikey, model, temperature, max_tokens)
    st.write(response)
elif user_input:
    st.warning("Please enter the OpenAI API Key in the side bar.")
else:
    st.write("Please provide the user input")


st.page_link("app.py", label="Home")
st.page_link("pages/wiki_arxiv.py", label="Chat with Search")
st.page_link("pages/rag_pdf_qa.py", label="RAG PDF Q&A")

