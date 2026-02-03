"""
Self-RAG System
---------------
A Self-Reflective Retrieval-Augmented Generation system using CrewAI, FastAPI, and IBM watsonx.

Features:
- Adaptive retrieval (only when needed)
- Relevance filtering of retrieved documents
- Self-critique and answer refinement
- Single agent architecture (efficient, 1 LLM call)
- AstraDB vector search with similarity scores
"""

# --- Imports ---
from dotenv import load_dotenv
import os
import json
import asyncio
from typing import Dict, List, Any
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from astrapy import DataAPIClient
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

# --- Load environment variables ---
load_dotenv()
WX_API_KEY = os.getenv("WATSONX_APIKEY")
PROJECT_ID = os.getenv("PROJECT_ID")
ASTRA_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_COLLECTION = os.getenv("ASTRA_DB_COLLECTION", "documents")
EXPECTED_API_KEY = os.getenv("ORCH_API_KEY")

# --- Connect to AstraDB ---
astra_client = DataAPIClient(ASTRA_TOKEN) if ASTRA_TOKEN else None
astra_db = astra_client.get_database_by_api_endpoint(ASTRA_ENDPOINT) if astra_client and ASTRA_ENDPOINT else None
astra_collection = astra_db.get_collection(ASTRA_COLLECTION) if astra_db else None

# --- Configure IBM watsonx LLM ---
llm = LLM(
    model="watsonx/meta-llama/llama-3-3-70b-instruct",
    base_url="https://us-south.ml.cloud.ibm.com",
    api_key=WX_API_KEY,
    project_id=PROJECT_ID,
    max_tokens=4000,
)

# --- FastAPI Setup ---
app = FastAPI(title="Self-RAG CrewAI", version="2.0")


class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: bool = False


# ============================================================================
# SELF-RAG TOOLS
# ============================================================================

class VectorSearchTool(BaseTool):
    """Performs semantic vector search with similarity scores."""

    name: str = "vector_search"
    description: str = "Search the knowledge base for relevant information. Returns documents with similarity scores. Use this when the query requires specific domain knowledge."

    def _run(self, query: str) -> str:
        if not astra_collection:
            return "Knowledge base not available."

        results = astra_collection.find(
            sort={"$vectorize": query},
            limit=5,
            include_similarity=True
        )

        output = []
        for doc in results:
            text = doc.get("text", "") or doc.get("content", "")
            similarity = doc.get("$similarity", 0)
            source = doc.get("metadata", {}).get("url", "")

            if text and similarity >= 0.5:  # Only include somewhat relevant docs
                entry = f"[Relevance: {similarity:.2f}] {text}"
                if source:
                    entry += f"\n(Source: {source})"
                output.append(entry)

        if not output:
            return "No relevant documents found in knowledge base."

        return "\n\n---\n\n".join(output)


class SelfCritiqueTool(BaseTool):
    """Evaluates the quality of a generated answer."""

    name: str = "self_critique"
    description: str = "Evaluate your draft answer for accuracy, completeness, and relevance. Use this to check if your answer needs improvement."

    def _run(self, draft_answer: str) -> str:
        critique_prompt = f"""
        Evaluate this answer critically:

        ANSWER: {draft_answer}

        Check for:
        1. Factual accuracy - Are claims supported?
        2. Completeness - Does it fully address the question?
        3. Clarity - Is it easy to understand?
        4. Relevance - Does it stay on topic?

        Respond with:
        - QUALITY: [GOOD/NEEDS_IMPROVEMENT]
        - ISSUES: [List any problems, or "None"]
        - SUGGESTION: [How to improve, or "None"]
        """

        result = llm.call(critique_prompt)
        return str(result)


# ============================================================================
# SELF-RAG AGENT
# ============================================================================

self_rag_agent = Agent(
    role="Self-RAG Assistant",
    goal="Provide accurate, helpful answers using adaptive retrieval when needed",
    backstory="""You are a helpful assistant. For domain-specific questions, search the knowledge base first.
If no relevant results are found, use your general knowledge to answer.
Always respond in the same language as the user's question.
Give direct, concise answers without explaining your reasoning process.""",
    llm=llm,
    tools=[VectorSearchTool(), SelfCritiqueTool()],
    verbose=False,
)


# ============================================================================
# SELF-RAG SYSTEM
# ============================================================================

class SelfRAGSystem:
    """Orchestrates the Self-RAG workflow."""

    def process_query(self, user_input: str) -> str:
        task = Task(
            description=f"""
            Answer this query: "{user_input}"

            Instructions:
            1. For domain-specific questions: Use vector_search first
            2. If search returns no results or low relevance: Answer from your general knowledge
            3. Always provide a complete, helpful answer in the user's language
            4. Do NOT include your reasoning process in the final answer
            5. Just give the direct answer to the user

            IMPORTANT: Your final output must be ONLY the answer itself, nothing else.
            """,
            agent=self_rag_agent,
            expected_output="A direct, helpful answer in the same language as the query",
        )

        crew = Crew(
            agents=[self_rag_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )

        result = crew.kickoff()
        return str(result)


def run_rag(query: str) -> str:
    """Convenience wrapper for the Self-RAG system."""
    return SelfRAGSystem().process_query(query)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/v1/chat")
async def chat_completion(request: ChatRequest, authorization: str = Header(None)):
    """Handle chat completions with Self-RAG."""

    if not EXPECTED_API_KEY or not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    provided = authorization.split("Bearer ")[1].strip()
    if provided != EXPECTED_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

    user_message = next((m for m in request.messages if m["role"] == "user"), None)
    if not user_message:
        raise HTTPException(status_code=400, detail="Missing user input")

    query = user_message["content"]

    if request.stream:
        return StreamingResponse(stream_response(query), media_type="text/event-stream")
    else:
        return {"response": run_rag(query)}


async def stream_response(query: str):
    """Stream RAG output as Server-Sent Events."""
    result_text = run_rag(query)
    words = result_text.split()
    chunk = []

    for i, word in enumerate(words, 1):
        chunk.append(word)
        if len(chunk) >= 25 or i == len(words):
            payload = {"chunk": " ".join(chunk)}
            yield f"data: {json.dumps(payload)}\n\n"
            chunk = []
            await asyncio.sleep(0.05)
    yield "data: [DONE]\n\n"


@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "Self-RAG System running 🚀", "version": "2.0"}


# --- Entry Point ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
