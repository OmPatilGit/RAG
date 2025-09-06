from models import GenModel
from InStorageRAG.vector import retriever
from InStorageRAG.prompts import INSTORAGE_RAG

model = GenModel()

while True:
    user = input("USER : ")
    if user.lower().strip() in ['bye', 'exit', 'quit', 'q']:
        break
    answer = retriever.invoke(user)
    prompt = INSTORAGE_RAG.format_prompt(context=answer[0].page_content, question=user)
    result = model.invoke(prompt)
    print("AI : ", result.content)

