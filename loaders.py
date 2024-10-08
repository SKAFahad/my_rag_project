import os
import re
import pandas as pd
import fitz  # PyMuPDF
import torch
from PIL import Image
from docx import Document as DocxDocument
from rank_bm25 import BM25Okapi
from transformers import BlipProcessor, BlipForConditionalGeneration

# Define custom loaders for different file types

class TextFileLoader:
    """Loader for text files."""
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_split(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        # Clean and trim the content
        cleaned_content = self.clean_and_trim(content)
        if len(cleaned_content) > 300:
            # Split the content into manageable chunks
            return self.split_text(cleaned_content)
        return []

    def clean_and_trim(self, text):
        # Remove extra spaces, newlines, and unwanted characters
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def split_text(self, text, chunk_size=1000, overlap=100):
        # Split text into chunks with overlap
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

class DocxFileLoader:
    """Loader for DOCX files."""
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_split(self):
        doc = DocxDocument(self.file_path)
        content = [p.text for p in doc.paragraphs]
        # Clean and trim the content
        cleaned_content = self.clean_and_trim(" ".join(content))
        if len(cleaned_content) > 300:
            return self.split_text(cleaned_content)
        return []

    def clean_and_trim(self, text):
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def split_text(self, text, chunk_size=1000, overlap=100):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

class XlsxFileLoader:
    """Loader for XLSX files."""
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_split(self):
        content = []
        xlsx = pd.ExcelFile(self.file_path)
        for sheet_name in xlsx.sheet_names:
            df = pd.read_excel(xlsx, sheet_name=sheet_name)
            content.append(df.to_string(index=False))
        # Clean and trim the content
        cleaned_content = self.clean_and_trim(" ".join(content))
        if len(cleaned_content) > 300:
            return self.split_text(cleaned_content)
        return []

    def clean_and_trim(self, text):
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def split_text(self, text, chunk_size=1000, overlap=100):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

class PyMuPDFLoader:
    """Loader for PDFs using PyMuPDF."""
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_split(self):
        content = []
        doc = fitz.open(self.file_path)
        for page in doc:
            content.append(page.get_text())
        # Clean and trim the content
        cleaned_content = self.clean_and_trim(" ".join(content))
        if len(cleaned_content) > 300:
            return self.split_text(cleaned_content)
        return []

    def clean_and_trim(self, text):
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def split_text(self, text, chunk_size=1000, overlap=100):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

class ImageFileLoader:
    """Loader for image files to generate captions using BLIP."""
    def __init__(self, file_path):
        self.file_path = file_path
        # Initialize the BLIP processor and model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

    def load_and_split(self):
        try:
            # Open the image file
            image = Image.open(self.file_path).convert('RGB')
            # Generate caption using BLIP
            text = self.generate_caption(image)
            # Clean and trim the content
            cleaned_content = self.clean_and_trim(text)
            if len(cleaned_content) > 0:
                # Since captions are short, we might not need to split
                return [cleaned_content]
            return []
        except Exception as e:
            print(f"Error processing image {self.file_path}: {e}")
            return []

    def generate_caption(self, image):
        # Prepare the image
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        # Generate the caption
        with torch.no_grad():
            out = self.model.generate(**inputs)
        caption = self.processor.decode(out, skip_special_tokens=True)
        return caption

    def clean_and_trim(self, text):
        text = re.sub(r'\s+', ' ', text).strip()
        return text
