# Context & System Memory — AI Productivity Agent v3.0

## 0. Phase Tracker
- Current Active Phase: **Phase 5 — Observability & Polish**
- Phase Progress:
  - [x] Phase 0 — Foundation
  - [x] Phase 1 — Core Tools
  - [x] Phase 2 — Agents
  - [x] Phase 3 — Hardening
  - [x] Phase 4 — API & HITL
  - [ ] Phase 5 — Observability & Polish

---

## 1. Progress Summary

### Just Completed (Phase 1)
- Created all 3 tool adapters: CalendarAdapter, EmailAdapter, ContestProvider
- Created all Pydantic schemas in `src/parsers.py` (LLM output schemas + DTOs)
- Created `src/tools/dateparser_util.py` (Improvement 1 — RELATIVE_BASE injection)
- Created `src/tools/calendar.py` (asyncio.Lock + safe_schedule, Improvement 2 foundation)
- Created `src/tools/email.py` (StubEmailAdapter with 3-email inbox)
- Created `src/tools/contests.py` (Levenshtein dedup, Improvement 5)
- 63/63 unit tests passing (pytest tests/unit/ -v)
- Root-cause fixes: missing `timezone` import in calendar.py, `datetime.utcnow()` deprecated → `datetime.now(timezone.utc)`

### Currently Working Features
- SQLiteCalendarAdapter: overlap query with configurable buffer, atomic safe_schedule
- AggregatingContestProvider: concurrent fetch + Levenshtein dedup
- dateparser_util: RELATIVE_BASE injection, returns None for unresolvable input
- StubEmailAdapter: 3-email hardcoded inbox (1 MEETING_REQUEST, 1 SPAM, 1 GENERAL)
- All Pydantic schemas: RoutingDecision, ContestSelectionResult, EmailClassification, DTOs

---

## 2. Current File Structure

```bash
d:\New folder (2)\ai-productivity-agent\
├── .env.example
├── .gitignore
├── alembic.ini
├── pytest.ini                  # asyncio_mode = auto
├── requirements.txt
├── context.md
├── agent_memory.db             # created by alembic upgrade head
├── src/
│   ├── __init__.py
│   ├── db.py                   # Event, EmailAction, AgentLog, PendingAction + async engine
│   ├── logger.py               # structlog JSON, configure_logging()
│   ├── parsers.py              # ALL Pydantic schemas and DTOs
│   └── tools/
│       ├── __init__.py
│       ├── dateparser_util.py  # parse(raw, user_now) — Improvement 1
│       ├── calendar.py         # CalendarAdapter + SQLiteCalendarAdapter (Lock + safe_schedule)
│       ├── email.py            # EmailAdapter + StubEmailAdapter
│       └── contests.py         # ContestProvider + Codeforces + LeetCode + Aggregating
├── migrations/
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│       ├── 0001_initial_events.py
│       ├── 0002_v2_add_end_time_source_external_id.py
│       └── 0003_v3_add_priority_unique_pending_actions.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # test_db (autouse), calendar_adapter, stub_email fixtures
│   └── unit/
│       ├── __init__.py
│       ├── test_dateparser_util.py   (11 tests — all pass)
│       ├── test_calendar_adapter.py  (16 tests — all pass)
│       ├── test_contest_provider.py  (14 tests — all pass)
│       └── test_parsers.py           (22 tests — all pass)
└── venv/
```

---

## 3. Environment & Runtime State

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite+aiosqlite:///./agent_memory.db` | Async runtime DB |
| `GOOGLE_API_KEY` | *(required)* | Gemini API |
| `HITL_EXPIRY_MINUTES` | `30` | HITL pending action TTL |
| `CALENDAR_TRAVEL_BUFFER_MINUTES` | `15` | Soft-conflict buffer |
| `DEFAULT_USER_TIMEZONE` | `UTC` | Fallback user timezone |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Model name |

```bash
# Run tests
pytest tests/unit/ -v

# Run all tests (when Phase 2+ complete)
pytest -v
```

---

## 4. API Contracts (ONLY implemented routes)

*None yet — Phase 4.*

---

## 5. Key Data Structures

### Active Pydantic Schemas (`src/parsers.py`)
| Schema | Purpose |
|---|---|
| `RoutingDecision` | Supervisor LLM output (agent, confidence 0–1) |
| `ContestSelectionResult` | Contest select node LLM output |
| `EmailClassification` | Per-email classify node LLM output (includes email_id, received_at, start, end) |
| `ContestDTO` | Contest data from providers |
| `EmailDTO` | Email data from adapters (includes received_at for ordering) |
| `ConflictInfo` | Calendar overlap result (includes buffer_applied) |

### Key Behavioral Facts
- `SQLiteCalendarAdapter.check_overlap()` reattaches `timezone.utc` to datetimes read from SQLite (naive → aware)
- `CalendarAdapter._lock` is class-level, shared across ALL instances (Improvement 2)
- `AggregatingContestProvider` deduplicates by Levenshtein distance ≤ 5 AND start time within 1 hour (Improvement 5)
- `parse(None, ...)` and `parse("", ...)` always return `None` — callers must handle

---

## 6. Dependency Map

| File | Imports From |
|---|---|
| `src/db.py` | `sqlmodel`, `sqlalchemy.ext.asyncio` |
| `src/logger.py` | `structlog` |
| `src/parsers.py` | `pydantic` only |
| `src/tools/dateparser_util.py` | `dateparser`, `src/logger.py` |
| `src/tools/calendar.py` | `src/db.py`, `src/parsers.py`, `src/logger.py` |
| `src/tools/email.py` | `src/parsers.py`, `src/logger.py` |
| `src/tools/contests.py` | `src/parsers.py`, `src/logger.py`, `httpx`, `Levenshtein` |
| `tests/conftest.py` | `src/db.py`, `src/tools/calendar.py`, `src/tools/email.py` |

---

## 7. Micro-Decisions & Constraints

| Decision | Rationale |
|---|---|
| `timezone.utc` reattached in `check_overlap` | SQLite strips tz info; added back when reading |
| `datetime.now(timezone.utc)` in `db.py` | `datetime.utcnow()` deprecated in Python 3.13 |
| Email agent uses `asyncio.gather` (not LangGraph Send) | Send-based fan-out complicates state management in LangGraph 1.1.6; asyncio.gather achieves same parallelism |
| All LLM calls via `src/llm_client.py` | Architecture rule: single LLM wrapper |
| LLM is injectable in all agents | Testability: tests pass mock LLM |

---

## 8. Next Immediate Steps (Phase 2 — Agents)

1. ✅ Create `src/llm_client.py` — singleton LLM wrapper
2. ✅ Create `src/supervisor.py` — route() + dispatch()
3. ✅ Create `src/agents/__init__.py`
4. ✅ Create `src/agents/contest.py` — 4-node LangGraph
5. ✅ Create `src/agents/email.py` — 3-node LangGraph (classify via asyncio.gather)
6. ✅ Create `src/hitl.py` — HITLManager stage/confirm
7. ✅ Create integration tests
8. Run `pytest tests/integration/ -v` — all must pass
