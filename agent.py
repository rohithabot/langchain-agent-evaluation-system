from pathlib import Path

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings


# ==========================================
# STEP 1: Initialize Ollama LLM
# ==========================================

local_llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# ==========================================
# STEP 2: Load knowledge base
# ==========================================

document_path = (
    Path(__file__).resolve().parent
    / "documents"
    / "agentic_ai.txt"
)

with open(document_path, "r", encoding="utf-8") as file:
    knowledge_text = file.read()

documents = [
    Document(page_content=knowledge_text)
]


# ==========================================
# STEP 3: Create vector store
# ==========================================

vector_store = InMemoryVectorStore.from_documents(
    documents=documents,
    embedding=embeddings
)


# ==========================================
# STEP 4: Define agent tools
# ==========================================

@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers and return the result."""
    return a * b


@tool
def text_analyzer(text: str) -> dict:
    """Count the number of words and characters in text."""
    return {
        "word_count": len(text.split()),
        "character_count": len(text)
    }


@tool
def knowledge_search(query: str) -> str:
    """Search the local knowledge base for information."""

    results = vector_store.similarity_search(
        query,
        k=2
    )

    return "\n\n".join(
        doc.page_content for doc in results
    )


@tool
def rag_search(question: str) -> str:
    """Retrieve relevant information using RAG."""

    results = vector_store.similarity_search(
        question,
        k=2
    )

    context = "\n\n".join(
        doc.page_content for doc in results
    )

    return (
        f"Relevant knowledge:\n{context}"
    )


# ==========================================
# STEP 5: Create LangChain agent
# ==========================================

agent_tools = [
    multiply_numbers,
    text_analyzer,
    knowledge_search,
    rag_search
]

agent = create_agent(
    model=local_llm,
    tools=agent_tools,
    system_prompt=(
        "You are a helpful Agentic AI assistant. "
        "Use multiply_numbers for multiplication. "
        "Use text_analyzer for word and character counting. "
        "Use knowledge_search for knowledge-base questions. "
        "Use rag_search when the user explicitly requests RAG. "
        "Do not invent information missing from the knowledge base."
    )
)


# ==========================================
# STEP 6: Test the agent
# ==========================================

if __name__ == "__main__":

    response = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "Calculate 125 multiplied by 48."
            }
        ]
    })

    print("\nAGENT RESPONSE:")
    print(response["messages"][-1].content)

    print("\nTOOLS USED:")

    for message in response["messages"]:

        for call in getattr(message, "tool_calls", []):

            print("Tool:", call["name"])
            print("Arguments:", call["args"])