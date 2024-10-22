import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_unstructured import UnstructuredLoader

from utils.langchain_funcs import get_qa_agent, get_retriever, set_tracer

load_dotenv()

if __name__ == "__page__":
    urls = st.text_area(
        "URLs",
        value="https://es.wikipedia.org/wiki/Universidad_Arturo_Prat, https://www.langchain.com/ ",
        height=100,
    )
    url_list = [url.strip() for url in urls.split(",")]

    docs = []

    for url in url_list:
        loader = UnstructuredLoader(web_url=url)
        docs.extend(loader.load())

    retriever = get_retriever(docs)
    agent = get_qa_agent(retriever)

    if question := st.chat_input():
        st.session_state.results = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={
                "callbacks": [set_tracer("Web Q&A")],
                "configurable": {"thread_id": st.session_state.thread_id},
            },
        )

    if not st.session_state.results:
        st.stop()

    for message_str in st.session_state.results["messages"]:
        if isinstance(message_str, HumanMessage):
            with st.chat_message("human"):
                st.write(message_str.content)
        elif isinstance(message_str, AIMessage):
            with st.chat_message("ai"):
                st.write(message_str.content)
