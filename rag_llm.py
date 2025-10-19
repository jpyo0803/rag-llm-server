from dotenv import load_dotenv

load_dotenv()

from langsmith import Client as LangSmithClient

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class RagLLM:
    def __init__(self, llm, retriever):
        hub_client = LangSmithClient()
        prompt = hub_client.pull_prompt("rlm/rag-prompt")

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
