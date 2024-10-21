import json

from pypdf import PdfReader

# Path to the updated PDF
pdf_path = "updated_docs/updated_sample-invoice.pdf"

# Open and read the PDF
with open(pdf_path, "rb") as file:
    reader = PdfReader(file)

    # Print the metadata
    metadata = reader.metadata
    print("Metadata for the PDF:")
    print(json.dumps(metadata, indent=4))
