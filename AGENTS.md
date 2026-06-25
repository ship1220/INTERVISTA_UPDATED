# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**Job PrepMate** — a FastAPI web application for AI-powered job interview preparation and course generation. Users select a job role/level, optionally upload a resume, then either take a voice-based mock interview with AI-generated questions and feedback, or generate a structured learning course. Uses Ollama (local) as primary LLM with Google Gemini as fallback.

## Build & Run Commands

```powershell
# Activate virtualenv
.\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt

# Initialize the SQLite database (creates jobprepmate.db)
python create_db.py

# Run the dev server
uvicorn main:app --reload
```

The app requires Ollama running locally. Gemini is an optional cloud fallback.
```
# Required — Ollama must be running
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=mistral     # default; also phi3 (3.8B, fastest) or llama3 (8B, most accurate)

# Optional — enables Gemini fallback
GEMINI_API_KEY=<key>
```
Other `.env` variables: `GEMINI_MODEL` (default `gemini-2.0-flash`).

There is **no test suite** in this project currently.

## Architecture

### Request Flow
`main.py` is the single FastAPI entrypoint. All routes live there: auth (signup/login via cookie), interview workflow, resume upload, and course generation. Routes call into `ai_service.py` for LLM interaction and use prompt templates from `prompts.py`.

### AI Service Layer (`ai_service.py`)
- `generate_content(prompt, *, use_cache, json_mode)` — tries Ollama `/api/chat` first (1 attempt, streaming), then falls back to Gemini if `GEMINI_API_KEY` is set (single attempt). Skips Gemini entirely when the key is missing. Has an in-memory TTL cache (5 min, max 200 entries). When `json_mode=True`, Ollama is configured with `format: "json"`, a JSON-only system prompt, and `temperature: 0.3`. Ollama calls use `num_predict: 1024` and `stream: True` (see streaming note below).
- `extract_json(text)` — robust JSON parser that strips markdown fences, fixes trailing commas, finds outermost `{}`/`[]`, and attempts to repair truncated JSON by closing unclosed brackets/braces. On failure, logs the full raw text for debugging.
- `evaluate_content(role, level, qa_list)` — async LLM-based content evaluation. Returns per-answer `score`, `feedback`, `strengths`, `weaknesses`, and `ideal_answer`. Also returns `overall_feedback` which is used as the performance summary to avoid a second LLM call. Uses `_normalize_response` to map variant AI key names and add default aggregate scores (`relevance_score`, `depth_score`, `star_method_score`).
- `generate_performance_summary(report_data)` — fallback LLM-generated summary, only called when `evaluate_content` doesn't return `overall_feedback`.
- `compute_overall_score(content_avg, clarity_avg, engagement_avg)` — weighted formula: 50% content + 30% clarity + 20% engagement.
- `compute_recruiter_verdict(overall_score, role)` — returns SHORTLISTED/BORDERLINE/REJECT with suitable roles.
- `analyze_speech_delivery(answer, duration)` — computes speaking pace, filler count, clarity score (0-100), engagement score (0-100).
- `compute_confidence_score(speech_analyses)` — aggregated confidence estimate from speech metrics.
- `evaluate_answers(role, qa_list)` — legacy convenience wrapper.

A reusable `httpx.AsyncClient` (`_get_ollama_client()`) is shared across Ollama requests. Gemini uses the `Client` API (new SDK) with automatic fallback to `GenerativeModel` (old SDK) and runs via `run_in_executor`.

### Data Layer
- `database.py` — SQLAlchemy engine and `SessionLocal` bound to `sqlite:///./jobprepmate.db`.
- `models.py` — ORM models: `User`, `UserTarget`, `SkillProgress`, `InterviewSession`, `InterviewAttempt`, `Course`, `Chapter`, `Unit`, `QuizAttempt`. Not all models are actively used by routes yet (e.g. `Course`, `Chapter`, `Unit`, `QuizAttempt` are defined but not wired up).
- DB dependency injected via `get_db()` generator in `main.py`.

### In-Memory State (not persisted across restarts)
- `interview_sessions` — maps username → current interview session (questions, answers, role, level).
- `resume_store` — maps username → extracted resume text.
- `report_store` — maps username → latest interview performance report dict.

### Frontend
Jinja2 templates in `templates/` with inline `<script>` blocks (no build step, no JS framework). Key interactions:
- **Interview page** (`interview.html`): uses Web Speech API (`SpeechRecognition` for STT, `SpeechSynthesisUtterance` for TTS) to conduct voice interviews. Fetches questions from `/api/interview/questions`, posts answers to `/api/interview/evaluate`.
- **Course page** (`course.html`): connects to `/api/course/stream` via Server-Sent Events (SSE) to progressively render course modules as the AI generates them.

### Prompt Engineering (`prompts.py`)
All LLM prompts are centralized here. Each function returns a formatted string. Prompts instruct the LLM to return pure JSON (no markdown fences), but `extract_json` handles non-compliance. When modifying prompts, keep the JSON schema examples in sync with the parsing/normalization logic in `main.py`.

**Prompt size discipline**: Prompts are kept compact to minimize time-to-first-token on CPU. Resume text is truncated to 800 chars, each answer to 200 chars in evaluation prompts. When editing prompts, do **not** add verbose instructions or long JSON schema examples — a one-line schema example is sufficient.

**Feedback format**: `content_analysis_prompt` returns structured feedback per question with five fields: `score`, `feedback` (2-3 sentence overview), `strengths` (what the candidate did well), `weaknesses` (gaps and areas to improve), and `ideal_answer` (3-4 sentence example of a strong answer). The `report.html` template renders all five fields. When modifying the evaluation schema, keep `_normalize_response` in `ai_service.py`, `detailed_answers` in `main.py`, and `report.html` in sync.

## Key Patterns to Follow

- **AI response normalization**: `main.py` normalizes variant key names from AI responses (e.g. `evaluations` → `feedback_per_question`, `better_answer` → `improved_answer`). When adding new AI-powered features, include similar normalization and fallback defaults.
- **json_mode**: All `generate_content` calls that expect JSON output should pass `json_mode=True`. This enables Ollama's `format: "json"` constraint and adjusts the system prompt.
- **Connection-error short-circuit**: `_try_ollama_with_retry` catches `ConnectError`/`ConnectTimeout`/`ConnectionRefusedError` and returns `None` immediately without retrying. This ensures fast fallback to Gemini when Ollama isn't running.
- **Fallback data**: Every AI call has a hardcoded fallback if parsing fails, so the UI always gets valid data.
- **Auth**: Cookie-based (`request.cookies.get("user")`), no JWT/session middleware. `get_current_user()` returns `None` for unauthenticated requests; routes redirect to `/login` or raise 401.
- **Passwords**: Truncated to 72 chars (bcrypt limit) before hashing.

## Performance / Latency Notes

- **Model choice matters**: Default is `mistral` (7B, fast tokenizer). Switch to `phi3` (3.8B) for fastest responses, or `llama3` (8B) for maximum accuracy on machines with GPU.
- **Streaming is critical**: `_ollama_chat` uses `stream: True`. With `stream: False`, Ollama buffers the entire response and won’t send any data until generation is complete — if generation takes longer than the httpx read-timeout, the request is killed even though Ollama is working. With streaming, each token resets the read-timeout, so total generation time is effectively unlimited. The 120s `OLLAMA_TIMEOUT_SECONDS` is a per-chunk timeout (only triggers if the model hangs and stops producing tokens).
- **num_predict**: Ollama calls cap output at 2048 tokens (increased to accommodate richer feedback with strengths/weaknesses/ideal_answer per question). The truncated-JSON recovery in `extract_json` (Attempt 5) can partially salvage responses that hit this cap.
- **Timeout chain**: Ollama uses streaming so total generation time is unbounded. The 120s timeout only fires if no tokens arrive for that long (model hung). If Ollama is unreachable, fallback to Gemini is immediate. Gemini has a 30s timeout.
- **Empty-response guard**: `api_interview_questions` explicitly checks for empty AI output before JSON parsing and falls back to static questions with a clear log message.
