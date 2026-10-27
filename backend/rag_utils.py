
from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 2,vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    prompt = f"""
    You are an AI assistant that helps AuroraStay Hotels front-desk and reservation staff answer guest questions about AuroraStay hotel policies (booking and cancellation terms, room and property information, refunds and additional charges, guest conduct, accessibility and complaint resolution).

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
1. Use only information explicitly stated in the retrieved policy information.
2. Do not use outside knowledge, assumptions, or general hotel-industry practices that are not present in the retrieved policy information.
3. Include all relevant details needed to answer the query completely, such as exact amounts, fees, time limits, deadlines, and the steps of any procedure.
4. Do not omit important conditions, exceptions, limitations, or eligibility rules (for example rate-plan, property-specific, or service-animal exceptions).
5. If the answer differs by property, rate plan, or room type, clearly state which property, rate plan, or room type each detail applies to.
6. Focus only on information directly relevant to the user's query and do not include unrelated policy details.
7. Answer the user's specific question directly in the first sentence, then add the supporting conditions.
8. Keep the answer concise, clear, and easy for hotel staff to read out to a guest.
9. Do not invent fees, charges, deadlines, refund amounts, room features, property names, or policy conditions.
10. If the retrieved information is insufficient, say:
"The available policy documents do not contain enough information to answer this question."
If the retrieved information conflicts, clearly mention the conflict.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 2,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
