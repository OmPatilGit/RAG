from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings

def locate_and_split(filepath : str):
    loader = PyPDFLoader(file_path=filepath)
    docs = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    chunks = text_splitter.split_documents(documents=docs)
    print("Chunking...")
    return chunks

def dense_embedding(content : str):
    model = OllamaEmbeddings(model="mxbai-embed-large:335m")
    embeddings = model.embed_query(content)
    print("Dense Embeddings...")
    return embeddings

def sparse_embedding(content : str):
    pass