# Agentic RAG System

Ein Multi-Agent RAG System mit Dual-LLM Verifikation, IBM watsonx und AstraDB.

> Agentic RAG erweitert klassisches RAG um spezialisierte, autonome Agenten die zusammenarbeiten.

---

## Architektur

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  RAG Agent (gpt-oss-120b)           │
│  - Recherchiert in Knowledge Base   │
│  - Generiert umfassende Antwort     │
│  - Optimiert für Kreativität        │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Verifier Agent (granite-4-h-small) │
│  - Prüft Faktentreue                │
│  - Erkennt Halluzinationen          │
│  - Optimiert für Präzision          │
└─────────────────────────────────────┘
    │
    ├─► PASS → Response an User
    │
    └─► FAIL → Feedback zur Verbesserung
```

---

## Was ist Agentic RAG?

```
Traditional RAG    →    Self-RAG           →    Agentic RAG
(Blind Retrieval)       (Self-Critique)         (Multi-Agent)
```

| Generation | Beschreibung | Limitation |
|------------|--------------|------------|
| **Traditional RAG** | Query → Retrieve → Generate | Keine Qualitätskontrolle |
| **Self-RAG** | Ein LLM kritisiert sich selbst | Gleicher Bias, eigene Fehler unsichtbar |
| **Agentic RAG** | Mehrere spezialisierte LLMs | Höhere Komplexität, mehr Kosten |

### Warum zwei verschiedene LLMs?

| Aspekt | Single-LLM | Dual-LLM |
|--------|------------|----------|
| **Bias** | Gleiche Fehler wiederholt | Unterschiedliche Perspektiven |
| **Halluzinationen** | Selbst-bestätigt | Unabhängig erkannt |
| **Training Data** | Identisch | Unterschiedlich |
| **Spezialisierung** | Allrounder | Generierung vs. Prüfung |

---

## Features

| Feature | Beschreibung |
|---------|--------------|
| **Adaptives Retrieval** | Sucht nur wenn nötig |
| **Relevanz-Filterung** | Nur Docs mit Score ≥ 0.5 |
| **Dual-LLM Verifikation** | Unabhängige Faktenprüfung |
| **RAG Agent** | OpenAI gpt-oss-120b (120B) |
| **Verifier Agent** | IBM Granite 4-H-Small (30B) |
| **AstraDB** | Vector-Suche mit $vectorize |

---

## Quickstart

### 1. Environment einrichten

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. `.env` Datei erstellen

```env
# IBM watsonx
WATSONX_APIKEY=<your-api-key>
PROJECT_ID=<your-project-id>

# AstraDB
ASTRA_DB_APPLICATION_TOKEN=<your-token>
ASTRA_DB_API_ENDPOINT=<your-endpoint>
ASTRA_DB_COLLECTION=<your-collection>

# API Security
ORCH_API_KEY=<your-api-key>
```

### 3. Server starten

```bash
python app.py
```

Server läuft auf `http://localhost:8000`

### 4. API testen

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Authorization: Bearer <ORCH_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"model": "test", "messages": [{"role": "user", "content": "Deine Frage"}]}'
```

---

## API Endpoints

| Endpoint | Methode | Beschreibung |
|----------|---------|--------------|
| `/` | GET | Health Check |
| `/v1/chat` | POST | Chat Completion |

---

## Implementierung

```python
# Zwei spezialisierte LLMs
generator_llm = LLM(model="watsonx/openai/gpt-oss-120b")
verifier_llm = LLM(model="watsonx/ibm/granite-4-h-small")

# RAG Agent: Recherchiert und generiert
rag_agent = Agent(
    role="RAG Research Assistant",
    llm=generator_llm,
    tools=[VectorSearchTool()],
)

# Verifier Agent: Prüft unabhängig
verifier_agent = Agent(
    role="Quality Assurance Reviewer",
    llm=verifier_llm,  # Anderes LLM!
    tools=[],
)

# Multi-Agent Crew
crew = Crew(
    agents=[rag_agent, verifier_agent],
    tasks=[rag_task, verify_task],
    process=Process.sequential,
)
```

---

## Agentic RAG Patterns

### Pattern 1: Generator + Verifier (Unser Ansatz)

```
Generator → Verifier → Output
```

### Pattern 2: Planner + Executor + Critic

```
Planner → Executor → Critic → Planner (Loop)
```

### Pattern 3: Specialized Experts

```
Router → Expert A / Expert B / Expert C → Aggregator
```

---

## Vergleich der Ansätze

| Aspekt | Traditional RAG | Self-RAG | Agentic RAG |
|--------|-----------------|----------|-------------|
| Retrieval | Immer | Adaptiv | Adaptiv |
| Verifikation | Keine | Selbst | Unabhängig |
| LLMs | 1 | 1 | 2+ |
| Halluzinationen | Häufig | Reduziert | Stark reduziert |
| Kosten | Niedrig | Niedrig | Mittel |
| Komplexität | Niedrig | Mittel | Hoch |

---

## Deployment (IBM Code Engine)

```bash
# Secrets erstellen
ibmcloud ce secret create --name agentic-rag-secrets \
  --from-literal WATSONX_APIKEY=... \
  --from-literal PROJECT_ID=... \
  --from-literal ASTRA_DB_APPLICATION_TOKEN=... \
  --from-literal ASTRA_DB_API_ENDPOINT=... \
  --from-literal ASTRA_DB_COLLECTION=... \
  --from-literal ORCH_API_KEY=...

# App deployen
ibmcloud ce app create --name agentic-rag \
  --build-source . \
  --env-from-secret agentic-rag-secrets
```

---

## Tech Stack

- **Framework:** CrewAI + FastAPI
- **RAG Agent:** OpenAI gpt-oss-120b (120B Parameter)
- **Verifier Agent:** IBM Granite 4-H-Small (30B Parameter)
- **Vector DB:** DataStax AstraDB
- **Deployment:** IBM Code Engine

---

## Weiterführende Ressourcen

- [Self-RAG Paper](https://arxiv.org/abs/2310.11511) - Grundlagen der Reflexion
- [CrewAI Documentation](https://docs.crewai.com/) - Multi-Agent Framework
