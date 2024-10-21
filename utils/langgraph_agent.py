from typing import Annotated, Sequence

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph.message import add_messages
from pypdf import PdfReader
from typing_extensions import TypedDict

LLM = ChatOpenAI(model="gpt-4o-mini")
EMBEDDINGS_MODEL = OpenAIEmbeddings(model="text-embedding-3-small")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def _get_tool(retriever):
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_pdf_docs",
        "Search and return information from PDF documents",
    )
    return [retriever_tool]


def load_uploaded_docs(uploaded_files):
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
    return docs


def get_retriever(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = InMemoryVectorStore.from_documents(
        documents=splits,
        embedding=EMBEDDINGS_MODEL,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever


def split_and_store(docs):
    pass
