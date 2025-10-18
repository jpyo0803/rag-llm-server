from rag_llm import RagLLM

if __name__ == "__main__":
    rag_llm = RagLLM()
    
    while True:
        query = input("Input: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = rag_llm.invoke(query)
        print("Response:", response)