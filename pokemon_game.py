from rag_llm import RagLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.llms import LlamaCpp

if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
 
    # llm = LlamaCpp(
    #     model_path="models/gemma-2-2b-it-Q4_K_M.gguf",
    #     n_ctx=4096,          # 문맥 길이 (필요 시 4096 가능)
    #     n_gpu_layers=32,     # 8GB GPU는 25~35 권장
    #     n_batch=256,         # VRAM 여유 있으면 512도 가능
    #     temperature=0.3,
    #     top_p=0.9,
    #     verbose=False
    # )

    # llm = LlamaCpp(
    #     model_path="models/Phi-3-mini-4k-instruct-q4.gguf",
    #     n_ctx=4096,          # 문맥 길이 (필요 시 4096 가능)
    #     n_gpu_layers=20,     # 8GB GPU는 25~35 권장
    #     n_batch=512,         # VRAM 여유 있으면 512도 가능  
    #     temperature=0.3,
    #     top_p=0.9,
    #     verbose=False
    # )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0
    )

    rag_llm = RagLLM(llm, retriever)
    
    query = "상대가 꼬부기를 냈을 때 나는 어떤 포켓몬을 내보내면 좋을까?"
    response = rag_llm.invoke(query)
    print("Response:", response)

    query = "상대가 파이리를 냈을 때 나는 어떤 포켓몬을 내보내면 좋을까?"
    response = rag_llm.invoke(query)
    print("Response:", response)

    query = "상대가 이상해씨를 냈을 때 나는 어떤 포켓몬을 내보내면 좋을까?"
    response = rag_llm.invoke(query)
    print("Response:", response)

    query = "상대가 피카츄를 냈을 때 나는 어떤 포켓몬을 내보내면 좋을까?"
    response = rag_llm.invoke(query)
    print("Response:", response)





