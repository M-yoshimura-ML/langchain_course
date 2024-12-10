import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

def create_tools():
    api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
    arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

    api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

    search = DuckDuckGoSearchRun(name="search")

    tools = [arxiv, wiki, search]
    return tools


def initialize_llm(api_key, model):
    llm = ChatGroq(api_key=api_key, model=model)
    return llm


st.title("Chat with Search")

with st.sidebar:
    st.title("Setting")
    api_key = st.text_input("Enter your Groq API Key", type="password")
    model = st.selectbox("select model", ['gemma2-9b-it', 'llama3-8b-8192', 'mixtral-8x7b-32768'])


if api_key:
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi, I'm chatbot who can search the web. How can I help you ?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="What is Machine Learning ?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

    llm = initialize_llm(api_key, model)
    tools = create_tools()
    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        input_dict = {"input": st.session_state.messages}
        response = search_agent.invoke(input_dict, {"callbacks": [st_cb]})

        response_content = response["output"]
        st.session_state.messages.append({"role": "assistant", "content": response_content})
        st.write(response_content)

else:
    st.warning("Please enter Groq API Key")

