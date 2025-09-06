import os
from dotenv import load_dotenv
from uuid import uuid4
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import PineconeHybridSearchRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def setup(filePath: str, indexName: str):
    if not filePath or not filePath.strip():
        return "Error: Filepath not specified."
    
    if not indexName or not indexName.strip():
        return "Error: Index name not specified."
    
    # Pinecone Models
    print("Initializing clients and models...")
    client = Pinecone(api_key=PINECONE_API_KEY)
    dense_embedder = OllamaEmbeddings(model="mxbai-embed-large:335m")
    
    # Create index if not present
    if indexName not in client.list_indexes().names():
        print(f"Creating index '{indexName}'...")
        client.create_index(
            name=indexName,
            dimension=1024,
            metric='dotproduct',
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not client.describe_index(indexName).status['ready']:
            time.sleep(1)
            
    index = client.Index(indexName)
    print("Pinecone index is ready.")

    # Load and chunk docs
    print(f"Loading and splitting document: {os.path.basename(filePath)}")
    loader = PyPDFLoader(file_path=filePath)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents=docs)
    print(f"Document split into {len(chunks)} chunks.")

    # Sparse Encoder
    print("Fitting BM25Encoder on the entire corpus...")
    corpus_text = [doc.page_content for doc in chunks]
    bm25_encoder = BM25Encoder()
    bm25_encoder.fit(corpus_text)

    vectors_to_upsert = []
    
    print("Generating dense and sparse embeddings for all chunks...")
    dense_embeddings = dense_embedder.embed_documents(corpus_text)
    sparse_embeddings = bm25_encoder.encode_documents(corpus_text)
    
    print("Preparing vectors for batch upsert...")
    for i, doc in enumerate(chunks):
        vectors_to_upsert.append({
            "id": str(uuid4()),
            "values": dense_embeddings[i],
            "sparse_values": sparse_embeddings[i],
            "metadata": {
                "text": doc.page_content,
                "source": os.path.basename(doc.metadata.get('source', 'Unknown')),
            }
        })

    print(f"Upserting {len(vectors_to_upsert)} vectors in batches...")
    for i in range(0, len(vectors_to_upsert), 100):
        batch = vectors_to_upsert[i:i+100]
        index.upsert(vectors=batch)

    print("Index setup complete and vectors have been upserted!")

    retriever = PineconeHybridSearchRetriever(embeddings=dense_embedder,sparse_encoder=bm25_encoder,index=index)
    print("Hybrid retriever initialized.")

    return f"Success! Index '{indexName}' is updated with the contents of {os.path.basename(filePath)}."

    