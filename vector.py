from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("game_reviews_clean.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location) or len(os.listdir(db_location)) == 0

vector_store = Chroma(
    collection_name="game_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        game = str(row["Game"]) if pd.notna(row["Game"]) else ""
        review = str(row["Review"]) if pd.notna(row["Review"]) else ""
        website = str(row["Website"]) if pd.notna(row["Website"]) else ""
        
        page_content = f"{game} {review} (Source: {website})".strip()
        
        document = Document(
            page_content=page_content,
            metadata={
                "score": row["Score"],
                "website": row["Website"]
            },
            id=str(i)
        )
        
        ids.append(str(i))
        documents.append(document)

    print(f"Total documents created: {len(documents)}")
    for doc in documents[:3]:
        print(doc.page_content)
        print(doc.metadata)
        print("---")

    vector_store.add_documents(documents, ids=ids)
    print("Documents added.")
    print("Total in vectorstore:", len(vector_store.get()["ids"]))
else:
    print("Existing vectorstore loaded.")
    print("Total in vectorstore:", len(vector_store.get()["ids"]))

retriever = vector_store.as_retriever(search_kwargs={"k": 10})

