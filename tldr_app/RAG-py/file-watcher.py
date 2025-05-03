import os
import json
import hashlib
import sys

import fitz  # PyMuPDF

def calculate_sha256(file_path):
    """Calculate SHA-256 checksum of a file."""
    try:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        return f"Error calculating SHA-256: {e}"

def get_pdf_title(pdf_path):
    """Extract the title from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        return metadata.get("title", "No title found")
    except Exception as e:
        return f"Error reading PDF: {e}"

def traverse_directory(directory, output_file):
    """Recursively traverse a directory, extract titles from PDFs, and save as JSON."""
    pdf_data = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                title = get_pdf_title(pdf_path)
                shasum = calculate_sha256(pdf_path)

                pdf_data.append({
                    "path": pdf_path,
                    "title": title,
                    "shasum": shasum
                })

    # Save output to a JSON file
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(pdf_data, json_file, indent=4)

    print(pdf_data)

if __name__ == "__main__":
    if len(sys.argv)<2:
        print('{"ERROR":"Empty directory"}')
        exit(1)

    output_file = "pdf_metadata.json"
    directory = sys.argv[1]

    if os.path.isdir(directory):
        traverse_directory(directory, output_file)
    else:
        print('{"ERROR":"Invalid Path"}')
        exit(1)