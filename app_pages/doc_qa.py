import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from streamlit_pdf_viewer import pdf_viewer

from utils.langchain_funcs import (
    get_qa_agent,
    get_retriever,
    load_uploaded_docs,
    set_tracer,
)

load_dotenv()


if __name__ == "__page__":
    with st.sidebar:
        st.session_state.uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
        )
        load_docs = st.button(
            "Load Documents",
            use_container_width=True,
            type="primary",
            disabled=not st.session_state.uploaded_files,
        )
        for uploaded_file in st.session_state.uploaded_files:
            pdf_viewer(uploaded_file.read(), width=700, height=400)

    if not st.session_state.uploaded_files and not load_docs:
        st.stop()

    docs = load_uploaded_docs(st.session_state.uploaded_files)
    retriever = get_retriever(docs)
    agent = get_qa_agent(retriever)

    if question := st.chat_input():
        st.session_state.results = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={
                "callbacks": [set_tracer("Doc Q&A")],
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
