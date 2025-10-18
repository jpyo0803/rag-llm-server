from dotenv import load_dotenv

load_dotenv()

from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langsmith import Client as LangSmithClient

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate

class RagLLM:
    def __init__(self):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever()

        hub_client = LangSmithClient()
        prompt = hub_client.pull_prompt("rlm/rag-prompt")

        # llm = LlamaCpp(
        #     model_path="models/gemma-2-2b-it-Q4_K_M.gguf",
        #     n_ctx=2048,          # 문맥 길이 (필요 시 4096 가능)
        #     n_gpu_layers=32,     # 8GB GPU는 25~35 권장
        #     n_batch=256,         # VRAM 여유 있으면 512도 가능
        #     temperature=0.3,
        #     top_p=0.9,
        #     verbose=False
        # )
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0
        )

        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])

        self.rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    def invoke(self, query):
        return self.rag_chain.invoke(query)
