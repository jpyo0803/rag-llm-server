import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

folder_path = "data/"
docs = []

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join(folder_path, filename), encoding='utf-8')
        docs.extend(loader.load())

print(f"문서 개수: {len(docs)}")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
splits = splitter.split_documents(docs)
print(f"분할된 문서 개수: {len(splits)}")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(splits, embedding=embeddings)
vectorstore.save_local("faiss_index")