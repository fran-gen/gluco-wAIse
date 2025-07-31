import os
import json
from typing import List

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OPENAI_API_KEY in .env file")


class KnowledgeBaseBuilder:
    def __init__(self, kb_path: str, save_dir: str, index_name: str = "index"):
        """
        Args:
            kb_path: Path to the JSON knowledge base file.
            save_dir: Directory where the FAISS index will be saved.
            index_name: Optional name for the FAISS index (default: "index").
        """
        self.kb_path = kb_path
        self.save_dir = save_dir
        self.index_name = index_name
        self.embeddings = OpenAIEmbeddings()

    def load_kb(self) -> List[Document]:
        """Loads the KB JSON and converts it into LangChain Documents."""
        with open(self.kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs = []
        for item in data:
            answer = item["answer"]
            for question in item["questions"]:
                docs.append(Document(page_content=question, metadata={"answer": answer}))
        return docs

    def build_and_save_index(self):
        """Builds FAISS vectorstore from KB and saves it."""
        print(f"[INFO] Loading KB from {self.kb_path}")
        docs = self.load_kb()

        print(f"[INFO] Generating embeddings for {len(docs)} documents...")
        vectorstore = FAISS.from_documents(docs, self.embeddings)

        os.makedirs(self.save_dir, exist_ok=True)
        vectorstore.save_local(folder_path=self.save_dir, index_name=self.index_name)

        print(f"[SUCCESS] Saved FAISS index to {self.save_dir}")


import shutil

def update_vectorstore_from_pdf(pdf_path: str, save_dir: str = "vectorstore/food_kb_index", index_name: str = "index"):
    """
    Parses a PDF file, chunks the text, embeds it, and saves a new FAISS index.
    This replaces the existing knowledge base.
    """
    print(f"[INFO] Loading PDF from {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    print(f"[INFO] Splitting PDF into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(pages)
     # Print first 2 chunks for debugging
    print(chunks[:2])

    print(f"[INFO] Generating embeddings for {len(chunks)} chunks...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Clear old FAISS index folder
    if os.path.exists(save_dir):
        print(f"[INFO] Clearing previous index at {save_dir}...")
        shutil.rmtree(save_dir)

    os.makedirs(save_dir, exist_ok=True)
    vectorstore.save_local(folder_path=save_dir, index_name=index_name)

    print(f"[SUCCESS] Replaced FAISS index at {save_dir}")


if __name__ == "__main__":
    builder = KnowledgeBaseBuilder(
        kb_path="gluco-wAIse/data/kb/diabetes_kb.json",
        save_dir="gluco-wAIse/vectorstore/food_kb_index",
        index_name="index"
    )
    builder.build_and_save_index()
