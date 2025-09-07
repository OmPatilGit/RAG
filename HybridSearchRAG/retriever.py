import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_ollama import OllamaEmbeddings
from pinecone_text.sparse import BM25Encoder


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


INDEX_NAME = "hybrid-rag-index"
FILE_PATH = r"D:\Project\RAG\docs\NIPS-2017-attention-is-all-you-need-Paper.pdf" 

def get_hybrid_retriever():
    print("Initializing clients and models...")
    client = Pinecone(api_key=PINECONE_API_KEY)
    index = client.Index(INDEX_NAME)
    dense_embedder = OllamaEmbeddings(model="mxbai-embed-large:335m")

    print(f"Loading corpus from {os.path.basename(FILE_PATH)} to fit the BM25Encoder...")
    loader = PyPDFLoader(file_path=FILE_PATH)
    corpus_docs = loader.load()
    corpus_text = [doc.page_content for doc in corpus_docs]
    bm25_encoder = BM25Encoder()
    bm25_encoder.fit(corpus_text)
    print("BM25Encoder is ready.")


    retriever = PineconeHybridSearchRetriever(
        embeddings=dense_embedder,
        sparse_encoder=bm25_encoder,
        index=index,
        top_k=2,
        text_key="text"
    )
    print("Hybrid retriever successfully created.")
    return retriever