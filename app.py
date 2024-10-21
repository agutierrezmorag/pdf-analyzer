import uuid

import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph

if __name__ == "__main__":
    st.set_page_config(layout="wide")

    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = None
    if "updated_files" not in st.session_state:
        st.session_state.updated_files = []
    if "extraction_df" not in st.session_state:
        st.session_state.extraction_df = None

    if "checkpointer" not in st.session_state:
        st.session_state.checkpointer = MemorySaver()
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "workflow" not in st.session_state:
        st.session_state.workflow = StateGraph(state_schema=MessagesState)

    metadata_extractor_page = st.Page(
        page="app_pages/metadata_extractor.py",
        icon=":material/assignment:",
        title="Metadata Extractor",
        default=True,
    )
    doc_qa_page = st.Page(
        page="app_pages/doc_qa.py",
        title="Document Q&A",
        icon=":material/question_answer:",
    )

    pg = st.navigation([metadata_extractor_page, doc_qa_page])
    pg.run()
