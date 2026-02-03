"""
Multi-Agent RAG System Template
-------------------------------
This template demonstrates how to build a modular multi-agent Retrieval-Augmented Generation (RAG)
system using CrewAI, FastAPI, and IBM watsonx.

Features:
- Multi-agent classification and routing
- Dynamic tool usage (query cleaning, classification, vector search)
- Streaming responses via Server-Sent Events (SSE)
- IBM watsonx LLM integration
- AstraDB vector search
- Extensible event listener system for debugging and live progress

Adapt and extend this for your own domain (e.g., manufacturing, finance, healthcare).
"""

# --- Imports ---
from dotenv import load_dotenv
import os
import re
import nltk
import queue
import threading
from typing import Dict, List, Any
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import time
import uuid
import asyncio
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from astrapy import DataAPIClient
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from crewai.events import BaseEventListener

# --- Setup NLTK for text preprocessing ---
# Downloads tokenizers and stopwords on first run if missing
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# --- Load environment variables ---
# These allow secure configuration via a .env file.
load_dotenv()
WX_API_KEY = os.getenv("WATSONX_APIKEY")
PROJECT_ID = os.getenv("PROJECT_ID")
ASTRA_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_COLLECTION = os.getenv("ASTRA_DB_COLLECTION", "documents")
EXPECTED_API_KEY = os.getenv("ORCH_API_KEY")

# --- Connect to AstraDB (for RAG retrieval) ---
# This assumes you've stored vector embeddings in an AstraDB collection.
astra_client = DataAPIClient(ASTRA_TOKEN) if ASTRA_TOKEN else None
astra_db = astra_client.get_database_by_api_endpoint(ASTRA_ENDPOINT) if astra_client and ASTRA_ENDPOINT else None
astra_collection = astra_db.get_collection(ASTRA_COLLECTION) if astra_db else None

# --- Configure IBM watsonx as the primary LLM ---
# This model handles reasoning, classification, and generation.
llm = LLM(
    model="watsonx/meta-llama/llama-3-3-70b-instruct",
    base_url="https://us-south.ml.cloud.ibm.com",
    api_key=WX_API_KEY,
    project_id=PROJECT_ID,
    max_tokens=4000,
)

# --- Initialise FastAPI ---
# Provides REST endpoints for querying the multi-agent system.
app = FastAPI(title="Multi-Agent CrewAI RAG", version="1.0")

# --- Define data schemas for API inputs ---
# These control validation and typing for chat and query requests.
class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: bool = False

class QueryRequest(BaseModel):
    query: str


# --- Define Tools ---
# Tools encapsulate specific actions that agents can perform.
# Each tool extends BaseTool and implements `_run()`.

class QueryCleanerTool(BaseTool):
    """Cleans and normalizes user queries for RAG search."""

    name: str = "query_cleaner"
    description: str = "Removes stopwords and noise, preserving short technical terms"
    keep_words: list = ["ac", "dc", "hv", "lv", "ip", "kw", "hp"]

    def _run(self, query: str) -> str:
        # Clean punctuation and lowercase text
        query = re.sub(r"[^a-zA-Z0-9\s]", "", query.lower())
        tokens = word_tokenize(query)
        # Keep important abbreviations or non-stopword terms
        filtered = [
            w for w in tokens if w in self.keep_words or (w not in stop_words and len(w) > 2)
        ]
        return " ".join(filtered) if filtered else query


class VectorSearchTool(BaseTool):
    """Performs semantic vector search using AstraDB."""

    name: str = "vector_search"
    description: str = "Retrieves relevant text chunks and metadata from AstraDB vector database"

    def _run(self, query: str) -> str:
        if not astra_collection:
            return "AstraDB not configured. Please set ASTRA_DB_* environment variables."

        # AstraDB vector search using $vectorize for automatic embedding
        results = astra_collection.find(
            sort={"$vectorize": query},
            limit=5,
            include_similarity=True
        )

        output = []
        for doc in results:
            text = doc.get("text", "") or doc.get("content", "")
            url = doc.get("metadata", {}).get("url", "")
            if text:
                output.append(f"{text}\n(Source: {url})" if url else text)
        return "\n\n".join(output) if output else "No relevant documents found."


class QueryClassifierTool(BaseTool):
    """Classifies incoming queries into different agent routes."""

    name: str = "query_classifier"
    description: str = "Classifies queries into 'knowledge_agent' or 'expert_agent'"

    def _run(self, query: str) -> str:
        # Prompt the LLM to determine which agent should handle the query
        prompt = f"""
        Classify the user query:
        - If it's general information or documentation → return "knowledge_agent"
        - If it requires reasoning, recommendations, or context → return "expert_agent"
        
        Query: {query}
        """
        result = llm.call(prompt)
        text = str(result).lower()
        return "knowledge_agent" if "knowledge_agent" in text else "expert_agent"


# --- Define Agents ---
# Agents are autonomous units powered by the LLM and optional tools.

supervisor_agent = Agent(
    role="Supervisor",
    goal="Classify incoming user queries and delegate tasks to appropriate agents",
    backstory="You are a routing specialist who analyzes queries and determines the best agent to handle them.",
    llm=llm,
    tools=[QueryClassifierTool()],
)

knowledge_agent = Agent(
    role="Knowledge Agent",
    goal="Answer factual and documentation-based queries using RAG",
    backstory="You are a knowledge retrieval specialist who finds and synthesizes information from the vector database.",
    llm=llm,
    tools=[QueryCleanerTool(), VectorSearchTool()],
)

expert_agent = Agent(
    role="Expert Agent",
    goal="Provide reasoning-based or context-aware responses using domain knowledge",
    backstory="You are a domain expert who provides in-depth analysis and reasoning-based answers.",
    llm=llm,
    tools=[VectorSearchTool()],
)


# --- Multi-Agent System Orchestration ---
# The system coordinates classification, routing, and generation.

class MultiAgentRAGSystem:
    def process_query(self, user_input: str) -> str:
        # Step 1: Classify query via the Supervisor agent
        classifier_task = Task(
            description=f"Classify the query: '{user_input}'",
            agent=supervisor_agent,
            expected_output="knowledge_agent or expert_agent",
        )

        supervisor_crew = Crew(agents=[supervisor_agent], tasks=[classifier_task], process=Process.sequential)
        classification = str(supervisor_crew.kickoff()).lower()

        # Step 2: Route query to appropriate agent
        if "knowledge_agent" in classification:
            task = Task(
                description=f"""
                Use query_cleaner → vector_search → synthesize clear answer for:
                "{user_input}"
                """,
                agent=knowledge_agent,
                expected_output="Concise, structured factual answer",
            )
            return str(Crew(agents=[knowledge_agent], tasks=[task], process=Process.sequential).kickoff())

        else:
            task = Task(
                description=f"""
                Use vector_search to generate a context-aware expert response for:
                "{user_input}"
                """,
                agent=expert_agent,
                expected_output="Technical or reasoning-based expert response",
            )
            return str(Crew(agents=[expert_agent], tasks=[task], process=Process.sequential).kickoff())


# --- Helper function for direct use ---
def run_rag(query: str):
    """Convenience wrapper for synchronous execution."""
    return MultiAgentRAGSystem().process_query(query)


# --- API Endpoint for Chat Completion ---
@app.post("/v1/chat")
async def chat_completion(request: ChatRequest, authorization: str = Header(None)):
    """Handles both standard and streaming chat completions."""

    # Basic API key authentication
    if not EXPECTED_API_KEY or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    provided = authorization.split("Bearer ")[1].strip()
    if provided != EXPECTED_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

    # Extract user input
    user_message = next((m for m in request.messages if m["role"] == "user"), None)
    if not user_message:
        raise HTTPException(status_code=400, detail="Missing user input")

    query = user_message["content"]

    # If stream=True, use Server-Sent Events
    if request.stream:
        return StreamingResponse(stream_response(query), media_type="text/event-stream")
    else:
        return {"response": run_rag(query)}


# --- Streaming Implementation ---
# This streams the model output to the client incrementally (chunked by words).
async def stream_response(query: str):
    """Streams RAG output as Server-Sent Events."""
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


# --- Health Check Endpoint ---
@app.get("/")
def health_check():
    """Simple health check for deployment verification."""
    return {"status": "ok", "message": "Multi-Agent RAG Template running 🚀"}


# --- Entry Point ---
# This runs the FastAPI server when executed directly.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


