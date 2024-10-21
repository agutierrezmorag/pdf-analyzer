import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from utils.langgraph_agent import LLM, get_retriever, load_uploaded_docs

load_dotenv()


if __name__ == "__page__":
    tracer = LangChainTracer(project_name="Doc Q&A")

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

    if not st.session_state.uploaded_files and not load_docs:
        st.stop()

    docs = load_uploaded_docs(st.session_state.uploaded_files)
    retriever = get_retriever(docs)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
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

    question_answer_chain = create_stuff_documents_chain(LLM, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    if question := st.chat_input():
        with st.chat_message("human"):
            st.write(question)
        results = rag_chain.invoke(
            {
                "input": question,
            },
            config={"callbacks": [tracer]},
        )

        with st.chat_message("ai"):
            st.write(results["answer"])
