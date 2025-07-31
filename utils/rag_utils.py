import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Correct relative path from agent.py to the FAISS index
vectorstore_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "vectorstore", "food_kb_index")
)

vectorstore = FAISS.load_local(
    folder_path=vectorstore_path,
    index_name="index",
    embeddings=OpenAIEmbeddings(),
    allow_dangerous_deserialization=True,
)

retriever = vectorstore.as_retriever()
