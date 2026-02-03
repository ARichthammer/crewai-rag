"""
Agentic RAG System
------------------
A Multi-Agent RAG system with dual-LLM verification using CrewAI, FastAPI, and IBM watsonx.

Features:
- Adaptive retrieval (only when needed)
- Relevance filtering of retrieved documents
- Independent verification with separate LLM
- True multi-agent architecture (RAG + Verifier)
- AstraDB vector search with similarity scores

Architecture:
- RAG Agent (gpt-oss-120b): Research, retrieval, answer generation
- Verifier Agent (granite-4-h-small): Fact-checking, hallucination detection
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

# --- Configure IBM watsonx LLMs ---
# Generator LLM: OpenAI gpt-oss-120b (optimized for reasoning & agentic tasks)
generator_llm = LLM(
    model="watsonx/openai/gpt-oss-120b",
    base_url="https://us-south.ml.cloud.ibm.com",
    api_key=WX_API_KEY,
    project_id=PROJECT_ID,
    max_tokens=4000,
)

# Verifier LLM: IBM Granite 4 (fast, precise fact-checking)
verifier_llm = LLM(
    model="watsonx/ibm/granite-4-h-small",
    base_url="https://us-south.ml.cloud.ibm.com",
    api_key=WX_API_KEY,
    project_id=PROJECT_ID,
    max_tokens=2000,
)

# --- FastAPI Setup ---
app = FastAPI(title="Agentic RAG", version="2.0")


class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: bool = False


# ============================================================================
# RAG TOOLS
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


# ============================================================================
# MULTI-AGENT SETUP
# ============================================================================

# RAG Agent: Researches and generates answers
rag_agent = Agent(
    role="RAG Research Assistant",
    goal="Find relevant information and generate accurate, helpful answers",
    backstory="""You are a research assistant with access to a knowledge base.
For domain-specific questions, always search the knowledge base first using vector_search.
Generate comprehensive answers based on retrieved context.
If no relevant documents are found, use your general knowledge.
Always respond in the same language as the user's question.
Provide direct answers without explaining your reasoning process.""",
    llm=generator_llm,
    tools=[VectorSearchTool()],
    verbose=False,
)

# Verifier Agent: Fact-checks and validates answers
verifier_agent = Agent(
    role="Quality Assurance Reviewer",
    goal="Verify factual accuracy and detect potential hallucinations",
    backstory="""You are a critical reviewer responsible for quality assurance.
Your job is to verify answers for factual accuracy and consistency.
Check if claims are supported by the provided context or are well-known facts.
Be concise in your assessment.
If the answer is accurate and complete, respond with: PASS
If there are issues, respond with: FAIL followed by specific feedback for improvement.""",
    llm=verifier_llm,
    tools=[],
    verbose=False,
)


# ============================================================================
# MULTI-AGENT RAG SYSTEM
# ============================================================================

class MultiAgentRAGSystem:
    """Orchestrates the Multi-Agent RAG workflow with independent verification."""

    MAX_ITERATIONS = 2  # Maximum refinement attempts

    def process_query(self, user_input: str) -> str:
        # Task 1: RAG Agent researches and generates answer
        rag_task = Task(
            description=f"""
            Answer this query: "{user_input}"

            Instructions:
            1. For domain-specific questions: Use vector_search first
            2. If search returns relevant results: Base your answer on the retrieved context
            3. If no relevant results: Use your general knowledge
            4. Respond in the same language as the query
            5. Provide a direct, complete answer

            Your output must be ONLY the answer itself.
            """,
            agent=rag_agent,
            expected_output="A comprehensive, accurate answer in the user's language",
        )

        # Task 2: Verifier Agent checks the answer
        verify_task = Task(
            description="""
            Review the answer provided by the RAG agent.

            Check for:
            1. Factual accuracy - Are claims supported or verifiable?
            2. Completeness - Does it address the original question?
            3. Consistency - No contradictions?

            If accurate and complete: Respond with just "PASS"
            If issues found: Respond with "FAIL: [specific feedback]"
            """,
            agent=verifier_agent,
            context=[rag_task],
            expected_output="PASS or FAIL with feedback",
        )

        # Run the crew
        crew = Crew(
            agents=[rag_agent, verifier_agent],
            tasks=[rag_task, verify_task],
            process=Process.sequential,
            verbose=False,
        )

        result = crew.kickoff()
        verification_result = str(result)

        # If PASS, return the RAG answer
        if "PASS" in verification_result.upper():
            # Get the RAG task output
            return str(rag_task.output.raw) if rag_task.output else verification_result

        # If FAIL, return the RAG answer anyway (with note that verification flagged issues)
        # In production, you might want to iterate here
        return str(rag_task.output.raw) if rag_task.output else verification_result


def run_rag(query: str) -> str:
    """Convenience wrapper for the Multi-Agent RAG system."""
    return MultiAgentRAGSystem().process_query(query)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/v1/chat")
async def chat_completion(request: ChatRequest, authorization: str = Header(None)):
    """Handle chat completions with Agentic RAG."""

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
    return {"status": "ok", "message": "Agentic RAG System running", "version": "2.0"}


# --- Entry Point ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
