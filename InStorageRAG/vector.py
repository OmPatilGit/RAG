import os
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

EMBEDDINGS_MODEL = OllamaEmbeddings(model="mxbai-embed-large:335m")

DB_LOCATION = "./chroma_db"
COLLECTION = "academics"

file_present = os.path.exists(DB_LOCATION)

def load_and_split(filePath : str, 
                   chunk_size : int = 500,
                   chunk_overlap : int = 100,):
    loader = PyPDFLoader(file_path=filePath)
    docs = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents=docs)
    return chunks

if not file_present:
    documents = []
    ids = []

    chunks = load_and_split(filePath=r"D:\Project\RAG\docs\the-illusion-of-thinking.pdf")
    
    for doc in chunks:
        document = Document(
            page_content=doc.page_content,
            metadata = {"timeStamp" : doc.metadata['moddate'],
                        "filePath" : os.path.basename(doc.metadata['source']),
                        "pageLabel" :doc.metadata['page_label'] },
            id = str(uuid4())
        )
        documents.append(doc)
        ids.append(id)

vector_store = Chroma(
    collection_name=COLLECTION,
    persist_directory=DB_LOCATION,
    embedding_function=EMBEDDINGS_MODEL
)

if not file_present:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs = {"k": 2}
)