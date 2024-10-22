import base64
import os
from collections import defaultdict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import TokenTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader, PdfWriter
from streamlit_pdf_viewer import pdf_viewer

from utils.langchain_funcs import (
    get_metadata_extraction_chain,
    load_uploaded_docs,
    set_tracer,
)

load_dotenv()


def displayPDF(uploaded_file):
    base64_pdf = base64.b64encode(uploaded_file.read()).decode("utf-8")

    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def merge_documents_by_source(docs):
    grouped_documents = defaultdict(list)
    for doc in docs:
        source = doc.metadata.get("source")
        grouped_documents[source].append(doc)

    merged_documents = []

    for source, docs in grouped_documents.items():
        merged_content = "\n".join([doc.page_content for doc in docs])
        merged_metadata = docs[0].metadata.copy()
        merged_document = Document(
            page_content=merged_content, metadata=merged_metadata
        )
        merged_documents.append(merged_document)

    return merged_documents


def document_metadata_to_pdf_metadata(extraction_data):
    pdf_metadata = {}

    for doc_metadata in extraction_data.metadata:
        if doc_metadata.autor:
            pdf_metadata["/Author"] = doc_metadata.autor
        if doc_metadata.empresas:
            pdf_metadata["/Subject"] = ", ".join(doc_metadata.empresas)
        if doc_metadata.keywords:
            pdf_metadata["/Keywords"] = ", ".join(doc_metadata.keywords)
        if doc_metadata.fechas_relevantes:
            pdf_metadata["/Fechas Relevantes"] = doc_metadata.fechas_relevantes

        # Custom metadata fields
        pdf_metadata["/Resumen de contenido"] = doc_metadata.resumen
        pdf_metadata["/ID Documento"] = doc_metadata.id_documento
        pdf_metadata["/Estatus"] = doc_metadata.status
        pdf_metadata["/Sensibilidad"] = doc_metadata.sensibilidad
        pdf_metadata["/Version"] = doc_metadata.version

    return pdf_metadata


if __name__ == "__page__":
    with st.sidebar:
        st.session_state.uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )
        for uploaded_file in st.session_state.uploaded_files:
            pdf_viewer(uploaded_file.read(), width=700, height=400)

    if not st.session_state.uploaded_files:
        st.stop()

    docs = load_uploaded_docs(st.session_state.uploaded_files)

    text_splitter = TokenTextSplitter(chunk_size=2000, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)

    extractor_chain = get_metadata_extraction_chain()
    st.session_state.extractions = extractor_chain.batch(
        texts, config={"callbacks": [set_tracer("Metadata Extractor")]}
    )

    extraction_data = []

    for extraction in st.session_state.extractions:
        for metadata in extraction.metadata:
            extraction_data.append(
                {
                    "id_documento": metadata.id_documento,
                    "fuente": metadata.fuente,
                    "resumen": metadata.resumen,
                    "empresas": ", ".join(metadata.empresas)
                    if metadata.empresas
                    else None,
                    "autor": metadata.autor,
                    "departamento": metadata.departamento,
                    "fechas_relevantes": metadata.fechas_relevantes,
                    "status": metadata.status,
                    "keywords": ", ".join(metadata.keywords)
                    if metadata.keywords
                    else None,
                    "sensibilidad": metadata.sensibilidad,
                    "version": metadata.version,
                }
            )

    extraction_df = pd.DataFrame(extraction_data)

    for i, row in extraction_df.iterrows():
        st.table(pd.DataFrame([row]))

        pdf_path = f"updated_docs/updated_{row['fuente'].split('/')[1]}"
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as file:
                st.download_button(
                    label=f"Download {os.path.basename(pdf_path)}",
                    data=file,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf",
                    key=f"download_{i}",
                    type="primary",
                    use_container_width=True,
                )

    output_dir = "updated_docs/"
    os.makedirs(output_dir, exist_ok=True)

    for doc, extraction in zip(texts, st.session_state.extractions):
        pdf_path = doc.metadata.get("source")

        if pdf_path:
            with open(pdf_path, "rb") as file:
                reader = PdfReader(file)
                writer = PdfWriter()

                for page in reader.pages:
                    writer.add_page(page)

                # Convert DocumentMetadata to PDF metadata format with mapping
                new_metadata = document_metadata_to_pdf_metadata(extraction)

                # Merge with existing metadata
                merged_metadata = {**reader.metadata, **new_metadata}

                # Add merged metadata to the writer
                writer.add_metadata(merged_metadata)

                # Construct updated file path
                updated_pdf_path = os.path.join(
                    output_dir, f"updated_{os.path.basename(pdf_path)}"
                )

                # Save the updated PDF
                with open(updated_pdf_path, "wb") as updated_file:
                    writer.write(updated_file)

                st.session_state.updated_files.append(updated_pdf_path)
