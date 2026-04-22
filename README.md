# AI Productivity Agent 

An advanced, agentic backend system designed to streamline your productivity by intelligently managing calendar scheduling and email triage. 

Built with **FastAPI**, **LangChain/LangGraph**, and **SQLModel** (async SQLite), this agent uses LLMs (Gemini) to process natural language requests, resolve conflicts, and stage high-stakes actions for human review.

---

## ⚡ Key Features

*   **Multi-Agent Architecture**: A Supervisor agent routes incoming requests to specialized sub-agents:
    *   **Contest Agent**: Fetches upcoming coding contests (Codeforces, LeetCode), deduplicates them based on Levenshtein distance, and schedules them.
    *   **Email Agent**: Triages inbox messages (Spam, Meeting Requests, General inquiries) and processes them in parallel.
*   **Human-in-the-Loop (HITL)**: High-stakes actions (like scheduling long events or deleting emails) are completely staged in the database as `PendingActions`. They are only executed after explicit user confirmation (`/agent/confirm/{id}`).
*   **Safe Concurrency**: Calendar interactions are guarded by class-level `asyncio.Lock` preventing scheduling race conditions or double-booking.
*   **Intelligent Scheduling**: Configurable travel buffers (e.g., ±15 mins) prevent soft conflicts when adding new calendar events.
*   **Observability**: Integrated structured JSON logging via `structlog`, complete request latency middleware, and a background task manager.

---

## 🛠️ Prerequisites

*   **Python**: 3.11 or higher
*   **Google Gemini API Key**: Get one from [Google AI Studio](https://aistudio.google.com/) (Free tier available).

---

## 🚀 Quick Start (Local Setup)

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/RecurringNoob/AI-productivity-agent.git
cd ai-productivity-agent

# Create and activate a virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\Activate.ps1
# Mac/Linux:
# source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configuration Options

Copy the example environment file and add your credentials:

```bash
cp .env.example .env
```

Open `.env` and configure accordingly. At minimum, you must set your `GOOGLE_API_KEY`:

```ini
# Required
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional Default Configurations
DATABASE_URL=sqlite+aiosqlite:///./agent_memory.db
LOG_LEVEL=INFO
HITL_EXPIRY_MINUTES=30
HITL_EXPIRY_CHECK_SECONDS=300
CALENDAR_TRAVEL_BUFFER_MINUTES=15
DEFAULT_USER_TIMEZONE=UTC
```

### 3. Database Initialization

The system uses [Alembic](https://alembic.sqlalchemy.org/en/latest/) for database migrations. Before running the server for the first time, you must initialize the `agent_memory.db`.

```bash
alembic upgrade head
```

### 4. Start the Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn src.api.main:app --reload --port 8000
```
*(Note for Production: Start with `--workers 1` due to the process-level `asyncio.Lock` inside the calendar adapter).*

---

## 🔌 API Reference

Once the server is running, you can view the fully interactive Swagger UI documentation at:
**[http://localhost:8000/docs](http://localhost:8000/docs)**

### Core Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Liveness probe indicating server is running. |
| `GET` | `/metrics` | Lightweight operational snapshot (Action counts). |
| `POST` | `/agent/run` | Dispatch a natural language instruction to the agent. |
| `GET` | `/agent/pending` | List all HITL actions currently awaiting your review. |
| `POST` | `/agent/confirm/{id}` | Confirm (`decision: "confirm"`) or undo (`decision: "undo"`) a pending action. |

### Example Usage `/agent/run`

```bash
curl -X POST http://localhost:8000/agent/run \
     -H "Content-Type: application/json" \
     -d "{\"user_input\": \"Find me a Codeforces contest this weekend\"}"
```

**Response Output:**
```json
{
  "response": "I found Codeforces Round 987 this Saturday. Since it's over 4 hours long, I've staged it for review. Please confirm.",
  "agent_used": "CONTEST_AGENT",
  "pending_action_id": "92a46038-d4bb-4adc-b5c9-bb81b5dcb4c9"
}
```

---

## 📁 Project Structure

```text
ai-productivity-agent/
├── src/
│   ├── api/
│   │   ├── main.py        # FastAPI App, Lifespan, Middleware & Routing
│   │   └── schemas.py     # Pydantic schemas for HTTP endpoints
│   ├── agents/
│   │   ├── contest.py     # Contest Scheduling LangGraph
│   │   └── email.py       # Email Triage LangGraph
│   ├── tools/
│   │   ├── calendar.py    # Atomic SQLiteCalendarAdapter
│   │   ├── contests.py    # Web scrapers/API wrappers for coding contests
│   │   └── email.py       # Interactors for reading/modifying inboxes
│   ├── db.py              # SQLModel schema definitions & Async session factory
│   ├── hitl.py            # Lifecycle management for Pending Actions
│   ├── llm_client.py      # Singleton wrapper for the ChatGoogleGenerativeAI
│   ├── supervisor.py      # LLM Routing gatekeeper logic
│   └── tasks.py           # Background asyncio loop tasks 
├── migrations/            # Alembic database migration scripts
├── tests/                 # 150+ Unit & Integration Tests (pytest)
├── alembic.ini
├── pytest.ini
└── requirements.txt
```

---

## 🧪 Testing

The codebase maintains 100% test coverage across complex logic via `pytest`, `pytest-asyncio`, and in-memory isolated database runs.

Run the entire test suite:
```bash
pytest tests/ -v
```

---

## 🚢 Deployment Note

If deploying to ephemeral PaaS providers (like Render or Railway), the SQLite database file (`agent_memory.db`) will be wiped on every deployment. For persistent production use, swap `DATABASE_URL` to point to a PostgreSQL instance and update the SQLAlchemy driver (`postgresql+asyncpg://...`). The structure is SQL-agnostic via SQLModel/Alembic.
