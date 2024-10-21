import os
from collections import defaultdict

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.tracers import LangChainTracer
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pypdf import PdfReader, PdfWriter

from utils.metadata_schema import ExtractionData

load_dotenv()


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
        if doc_metadata.fecha_reunion:
            pdf_metadata["/CreationDate"] = doc_metadata.fecha_reunion

        # Custom metadata fields
        pdf_metadata["/ID Documento"] = doc_metadata.id_documento
        pdf_metadata["/Estatus"] = doc_metadata.status
        pdf_metadata["/Sensibilidad"] = doc_metadata.sensibilidad
        pdf_metadata["/Version"] = doc_metadata.version

    return pdf_metadata


if __name__ == "__page__":
    st.write("Metadata Extractor")
    tracer = LangChainTracer(project_name="Metadata Extractor")
    loader = DirectoryLoader("docs/", loader_cls=PyPDFLoader, glob="*.pdf")
    docs = loader.load()

    merged_documents = merge_documents_by_source(docs)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un experto en extraer metadatos desde una variedad de documentos PDF empresariales. "
                "Tu tarea es analizar el documento proporcionado y extraer todos los metadatos disponibles. "
                "En caso de que no puedas extraer un metadato, simplemente omítelo.",
            ),
            ("human", "{text}"),
        ]
    )
    extractor_chain = prompt | llm.with_structured_output(
        schema=ExtractionData,
        include_raw=False,
    )
    extractions = extractor_chain.batch(
        merged_documents, config={"callbacks": [tracer]}
    )

    extracted_data = []

    for extraction in extractions:
        st.write(extraction)
        extracted_data.extend(extraction.metadata)

    st.divider()

    output_dir = "updated_docs/"
    os.makedirs(output_dir, exist_ok=True)

    for doc, extraction in zip(docs, extractions):
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
