# Agentic RAG Agent – Code Engine Deployment

Vollständiges CLI-basiertes Deployment des Agentic RAG Multi-Agent Systems auf IBM Cloud Code Engine.

## Aktuelle Deployment-Werte

| Einstellung | Wert |
|-------------|------|
| **App URL** | https://agentic-rag.252q85btszgp.us-south.codeengine.appdomain.cloud |
| Projekt | ce-itz-wxo-69661f6ca47c0c599d6689 |
| Resource Group | ce-itz-wxo-69661f6ca47c0c599d6689 |
| App Name | agentic-rag |
| Region | us-south |
| Secret Name | agentic-rag-secrets |
| Registry Secret | ce-icr-secret |
| Min Scale | 1 (kein Cold Start) |
| Max Scale | 1 |
| **Auth** | Keine (auth_scheme: NONE) |

## Architektur

Die App nutzt CrewAI mit zwei Agenten:
- **RAG Agent**: Sucht in AstraDB (Datastax) nach relevanten Dokumenten
- **Verifier Agent**: Prüft und verifiziert die Antworten

API-Endpoint: `/v1/chat/completions` (OpenAI-kompatibel)

## Voraussetzungen

- IBM Cloud Account
- IBM Cloud CLI: `brew install ibmcloud-cli`
- Plugins: `ibmcloud plugin install code-engine container-registry`

## Quick Deploy (wenn alles eingerichtet ist)

```bash
# 1. Login & Projekt wählen
ibmcloud login --sso
ibmcloud target -r us-south -g ce-itz-wxo-69661f6ca47c0c599d6689
ibmcloud ce project select --name ce-itz-wxo-69661f6ca47c0c599d6689

# 2. In Projektverzeichnis wechseln
cd "/Users/alexanderrichthammer/Downloads/Visual Code Repo/watsonx/crewai-agent"

# 3. App aktualisieren
ibmcloud ce app update --name agentic-rag --build-source .

# 4. Status prüfen
ibmcloud ce app get --name agentic-rag
```

---

## Erstmaliges Setup (Schritt für Schritt)

### 1. Login & Targeting

```bash
ibmcloud login --sso
ibmcloud target -r us-south
ibmcloud target -g ce-itz-wxo-69661f6ca47c0c599d6689
```

### 2. Code Engine Projekt auswählen

```bash
ibmcloud ce project select --name ce-itz-wxo-69661f6ca47c0c599d6689
```

### 3. Secrets prüfen/erstellen

Prüfen welche Secrets existieren:
```bash
ibmcloud ce secret list
```

Falls `agentic-rag-secrets` fehlt:
```bash
ibmcloud ce secret create \
  --name agentic-rag-secrets \
  --from-literal WATSONX_APIKEY=<DEIN_WATSONX_API_KEY> \
  --from-literal PROJECT_ID=<DEIN_WATSONX_PROJECT_ID> \
  --from-literal ASTRA_DB_APPLICATION_TOKEN=<DEIN_ASTRA_TOKEN> \
  --from-literal ASTRA_DB_API_ENDPOINT=<DEIN_ASTRA_ENDPOINT> \
  --from-literal ASTRA_DB_COLLECTION=ai_agent_strategy
```

Falls `ce-icr-secret` fehlt:
```bash
ibmcloud iam api-key-create ce-icr-key -d "Code Engine ICR access"
# API-Key sofort sichern!

ibmcloud ce registry create \
  --name ce-icr-secret \
  --server us.icr.io \
  --username iamapikey \
  --password <ICR_API_KEY>
```

### 4. App erstellen (erstmaliges Deployment)

```bash
cd "/Users/alexanderrichthammer/Downloads/Visual Code Repo/watsonx/crewai-agent"

ibmcloud ce app create \
  --name agentic-rag \
  --build-source . \
  --build-strategy dockerfile \
  --build-size large \
  --registry-secret ce-icr-secret \
  --env-from-secret agentic-rag-secrets \
  --port 8080 \
  --min-scale 1 \
  --max-scale 1 \
  --memory 4G \
  --cpu 1
```

### 5. URL abrufen & testen

```bash
# URL anzeigen
ibmcloud ce app get --name agentic-rag --output url

# Health Check
curl https://agentic-rag.252q85btszgp.us-south.codeengine.appdomain.cloud/

# Chat Completion
curl -X POST https://agentic-rag.252q85btszgp.us-south.codeengine.appdomain.cloud/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"agentic-rag","messages":[{"role":"user","content":"Hello"}]}'
```

---

## App aktualisieren (nach Code-Änderungen)

```bash
cd "/Users/alexanderrichthammer/Downloads/Visual Code Repo/watsonx/crewai-agent"
ibmcloud ce app update --name agentic-rag --build-source .
```

---

## Authentifizierung

Die App verwendet **keine Authentifizierung** (`auth_scheme: NONE`).

| Endpoint | Beschreibung |
|----------|--------------|
| `GET /` | Health Check |
| `GET /health` | Alternative Health Check |
| `POST /v1/chat/completions` | Chat Completion (OpenAI-kompatibel) |
| `OPTIONS /v1/chat/completions` | CORS Preflight |

> **Hinweis:** Für watsonx Orchestrate ist `auth_scheme: NONE` der einfachste Ansatz.
> Für Produktionsumgebungen kann Connection-basierte Auth verwendet werden.

---

## Skalierung ändern

**Aktuelle Konfiguration:** min-scale=1, max-scale=1 (immer genau 1 Instanz)

```bash
# Immer genau 1 Instanz (Standardkonfiguration für diese App)
ibmcloud ce app update --name agentic-rag --min-scale 1 --max-scale 1

# Kosten sparen (Cold Start möglich, skaliert auf 0 wenn idle)
ibmcloud ce app update --name agentic-rag --min-scale 0 --max-scale 1
```

---

## watsonx Orchestrate Integration

1. `agentic_rag_agent.yaml` prüfen/anpassen:
   ```yaml
   spec_version: v1
   kind: external
   name: agentic_rag
   title: Agentic RAG Agent powered by Datastax
   nickname: agentic_rag
   provider: external_chat
   description: |
     Use this agent for AI strategy and agentic AI topics.
   tags: [rag, multi-agent, verification]
   api_url: "https://agentic-rag.252q85btszgp.us-south.codeengine.appdomain.cloud/v1/chat/completions"
   auth_scheme: NONE
   chat_params:
     model: agentic-rag
     stream: true
   config:
     hidden: false
     enable_cot: false
   ```

   **Wichtige Punkte:**
   - `api_url` muss den vollen Pfad inkl. `/v1/chat/completions` enthalten
   - `auth_scheme: NONE` ist der einfachste Ansatz
   - `model` und `stream` in `chat_params` sind erforderlich

2. Agent importieren:
   ```bash
   orchestrate agents import -f agentic_rag_agent.yaml
   ```

3. Agent entfernen (falls nötig):
   ```bash
   orchestrate agents remove -n agentic_rag -k external
   ```

---

## Troubleshooting

```bash
# App-Logs anzeigen
ibmcloud ce app logs --name agentic-rag

# Build-Logs anzeigen
ibmcloud ce buildrun list
ibmcloud ce buildrun logs --name <BUILDRUN_NAME>

# App-Status detailliert
ibmcloud ce app get --name agentic-rag

# App löschen
ibmcloud ce app delete --name agentic-rag

# Secret aktualisieren
ibmcloud ce secret delete --name agentic-rag-secrets
ibmcloud ce secret create --name agentic-rag-secrets --from-literal ...
```

### Häufige Fehler

| Fehler | Ursache | Lösung |
|--------|---------|--------|
| HTTP 405 Method Not Allowed | CORS fehlt oder api_url unvollständig | CORS Middleware prüfen, `/v1/chat/completions` in api_url |
| HTTP 422 Validation Error | Pydantic zu strikt | Flexibles `req.json()` statt Pydantic |
| Agent antwortet nicht | `model` fehlt in chat_params | `model` in YAML chat_params hinzufügen |
| "This action is forbidden" | Session abgelaufen | `ibmcloud login --sso` |
| Build zu groß | .venv wird hochgeladen | `.ceignore` prüfen |

---

## Wichtige Dateien

| Datei | Beschreibung |
|-------|--------------|
| `app.py` | FastAPI-Anwendung mit CORS Middleware |
| `Dockerfile` | Container-Build-Definition |
| `.dockerignore` | Ausgeschlossene Dateien beim Container-Build |
| `.ceignore` | Ausgeschlossene Dateien beim Code Engine Upload |
| `agentic_rag_agent.yaml` | watsonx Orchestrate Agent-Definition |
| `requirements.txt` | Python-Abhängigkeiten |

---

## Hinweise

- **CORS**: Die App hat CORS Middleware für watsonx Orchestrate (OPTIONS Preflight Requests)
- **Cold Start**: Mit min-scale=1 läuft immer eine Instanz (kein Cold Start)
- **URL-Änderung**: Falls die URL sich ändert, muss `agentic_rag_agent.yaml` angepasst und der Agent neu importiert werden
