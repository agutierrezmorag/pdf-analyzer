import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()


if __name__ == "__page__":
    tracer = LangChainTracer(project_name="Doc Q&A")
    llm = ChatOpenAI(model="gpt-4o-mini")

    with st.sidebar:
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )
        load_docs = st.button(
            "Load Documents",
            use_container_width=True,
            type="primary",
            disabled=not uploaded_files,
        )

    if not uploaded_files and not load_docs:
        st.stop()
    docs = []
    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
        content = ""
        for page in reader.pages:
            content += page.extract_text()

        doc = Document(
            page_content=content, metadata={"source": f"docs/{uploaded_file.name}"}
        )
        docs.append(doc)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = InMemoryVectorStore.from_documents(
        documents=splits, embedding=OpenAIEmbeddings(model="text-embedding-3-small")
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

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

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
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
