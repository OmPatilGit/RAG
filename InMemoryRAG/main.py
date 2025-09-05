from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import retrieval_qa
from langchain_community.vectorstores import FAISS 
from langchain_ollama import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from models import EmbModel, GenModel
from prompts import INMEMORY_RETRIEVER


def setup(file_path : str):
    loader = PyPDFLoader(file_path=file_path)
    docs = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200)
    chunks =  text_splitter.split_documents(docs)
    
    embeddings = OllamaEmbeddings(model = "mxbai-embed-large:335m")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever()

    llm = GenModel()

    chain = (
        {"context" : retriever, "question" : RunnablePassthrough()}
        | INMEMORY_RETRIEVER
        | llm
        | StrOutputParser() 
    )
    return chain

if __name__ == "__main__":
    qa_chain = setup(r"D:\Project\RAG\docs\the-illusion-of-thinking.pdf")

    while True:
        user = input("USER : ")
        if user.lower().strip() in ['exit', 'bye']:
            break

        answer = qa_chain.invoke(user)
        print("AI : ", answer)


