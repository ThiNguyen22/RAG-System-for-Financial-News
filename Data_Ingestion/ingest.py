from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document
from typing import List
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

df = pd.read_pickle("final_data.pkl")

embeddings = HuggingFaceEmbeddings(
    model_name="multi-qa-mpnet-base-dot-v1", model_kwargs={"device": "cpu"}
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
)

raw_texts: List[Document] = []
for _, row in df.iterrows():
    if not isinstance(row["title_en"], str) or not isinstance(row["text_en"], str):
        continue
    
    metadata = {
        k: v
        for k, v in row.items()
        if k in ["text", "title", "title_en", "url", "new_id", "last_update"]
    }
    
    raw_texts.append(Document(page_content=row["text_en"], metadata=metadata))

docs = text_splitter.split_documents(raw_texts)
unique_docs = []
exist_text = set()

for i in docs:
    if len(i.page_content.split()) < 5:
        continue
    if i.page_content not in exist_text:
        unique_docs.append(i)
        exist_text.add(i.page_content)
    else:
        continue
print("total docs: ", len(unique_docs))

vectorstore = FAISS.from_documents(unique_docs, embeddings)

vectorstore.save_local("vectorstore")
print("Saving vector: ...")
