# localAIv9

**Intent-to-Software Pipeline** — Python-native runtime, JSON-native contracts, deterministic architecture, mathematically grounded.

> **LLM = Idea Engine. Python = Structure Engine.**
> The LLM generates maximum creative potential. Python ingests, normalizes, prunes, and enforces contracts. No stubs. No placeholders. Real results.

---

## Quick Start

```bash
# Install dependencies
pip install pydantic httpx nicegui

# Run the full pipeline with a custom prompt
python localAIv9.py "Build a task manager with drag-and-drop boards"

# Launch the interactive dashboard
python localAIv9.py --ui

# Execute generated tasks (cron runner)
python localAIv9.py --execute

# Force-complete a specific task
python localAIv9.py --accelerate T001
```

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_HOST` | `http://172.30.80.1:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `gemma4:26b` | Model to use for LLM calls |
| `CRON_FACTOR` | `20` | Multiplier for task delay in cron schedule |

---

## Architecture

### 13-Section Pipeline (S0–S12)

| Section | Name | Type | Description |
|---------|------|------|-------------|
| **S0** | Runtime Gate Check | Gate | Validates imports, Ollama reachability, model availability. Fails fast before any pipeline step. |
| **S1** | Intent Extraction | LLM | Converts natural language prompt into structured product spec (features, entities, actions, architecture, constraints, success criteria). |
| **S2** | UI Structure Generation | LLM | Generates screen/component hierarchy (pages, modals, tabs) — pure JSON, no HTML, no styling. |
| **S3** | Theme Derivation | LLM + Web | Derives visual theme from 6 DuckDuckGo web searches (design trends, color palettes) + product spec + UI structure. WCAG AA contrast compliance. |
| **S4** | Feature DAG Mapping | Deterministic | Maps features to screens/components, builds action DAG, validates completeness. Self-corrects broken layers. |
| **S5** | DAG Pruning | Deterministic | Removes orphan nodes (no dependencies AND no dependents). |
| **S6** | Task Scheduling | Deterministic | Converts pruned DAG into scheduled tasks with sequential IDs (T001…), layer assignment, and time estimates. |
| **S7** | System Integration | Deterministic | Integrates themed screens with task DAG. Validates routing and theme consistency. |
| **S8** | Export & Cron Schedule | Deterministic | Generates all file artifacts: task scaffolds, cron schedule, task/test manifests, export artifact. |
| **S9** | Orchestration & Logging | Runtime | Pipeline execution, task runner with self-repair, cron scheduler, task acceleration, JSONL telemetry. |
| **S10** | Fine-Tune Preparation | Deterministic | Produces instruction/output dataset from session telemetry. Detects torch/unsloth availability. |
| **S11** | Dashboard Control Plane | Status | Reports dashboard mode, endpoints, task-control capabilities. |
| **S12** | Final Aggregation | Status | Pure status snapshot — task counts, artifact keys, finetune readiness, session ID. |

### Pipeline Flow

```
S0 (Gate) → S1 (Intent) → S2 (UI) → S3 (Theme) → S4 (DAG Map)
    → S5 (Prune) → S6 (Schedule) → S7 (Integrate) → S8 (Export)
    → S10 (Finetune) → S11 (Dashboard) → S12 (Aggregate)
```

S9 is the orchestration layer — it runs the pipeline, not a sequential step.

---

## Key Capabilities

### Gate Check (S0)
- Runs at import time before any pipeline code executes
- Validates: Python imports, Ollama host connectivity, model availability
- Fails fast with actionable error messages — pipeline never starts on bad runtime

### LLM Normalization Layer
- **Auto-corrects missing fields**: Features without `layer` get `"fullstack"` assigned
- **Auto-casts types**: Theme dict values coerced to strings regardless of LLM output type
- **JSON extraction**: Strips markdown fences, finds first `{...}` block in raw LLM response
- **Response caching**: MD5-keyed cache in `.llm_cache/` prevents redundant LLM calls

### Self-Correction & Repair
- **S1/S2/S3**: Invalid JSON triggers recovery prompt with stricter instructions. S3 has up to 2 retries then falls back to deterministic `_fallback_theme()`.
- **S4**: Validation failures trigger targeted regeneration — re-runs only the broken layer (function1 for API issues, function2 for UI mapping issues).
- **S9 (`build_one_task`)**: Task execution failures trigger LLM-based code repair — sends error message + current code to LLM, extracts repaired Python, re-executes.

### Web Search Integration (S3)
- `_web_search(query)` scrapes DuckDuckGo HTML for result snippets
- 6 distinct searches: UX trends, UI trends, award palettes, skin tone palettes, makeup shades, accent colors
- Results fed into LLM context for data-driven theme derivation

### DAG Scheduling (S4–S6)
- Deterministic feature-to-screen mapping via string matching
- Sequential DAG construction from actions
- Orphan node pruning (graph analysis)
- Layer-based time estimation: frontend=30s, backend=45s, fullstack=60s

### Cron Runner (S9)
- Dependency-aware parallel task execution
- Waits for all dependencies to complete before starting each task
- Applies `delay_seconds = est_seconds × CRON_FACTOR`
- Per-task file locks (`locks/` directory) with 120-second timeout

### Task Acceleration
- `accelerate(task_id)` instantly marks a task as "complete" in `tasks.json`
- CLI: `python localAIv9.py --accelerate T001`
- Dashboard: "Accelerate" button with task ID input

### Fine-Tune Preparation (S10)
- Reads `training_data/stream.jsonl`, filters out error records
- Produces `output/finetune_dataset.jsonl` in instruction/output JSONL format
- Produces `output/training_manifest.json` with torch/unsloth availability and readiness flag

### Live Dashboard (S11 + `start_ui`)
- NiceGUI web UI at `127.0.0.1:8080`
- **Interactive controls**:
  - Pipeline prompt textarea (custom prompts)
  - "Run Pipeline" button (executes full S1–S12 pipeline)
  - "Execute Scheduled Tasks" button (runs cron runner)
  - "Refresh Tasks" button (manual table refresh)
  - Task table with ID, Title, Status, Depends On
  - "Accelerate Task" input + button (force-complete any task)
  - Live log area (pipeline progress, errors, timestamps)
- Auto-refreshes task table every 3 seconds, flushes log every 1 second
- Falls back to "mirror-only" mode when NiceGUI is not installed

### Telemetry-Driven
- Every pipeline step logged with timing, inputs, outputs, tool calls
- Per-session JSONL logs: `session_{uuid}.jsonl`
- Continuous training stream: `training_data/stream.jsonl` (cumulative, quality-filtered)

---

## File Artifacts

| File | Section | Description |
|------|---------|-------------|
| `.llm_cache/{md5hash}.json` | S2 | Cached LLM responses (keyed by host+model+temp+tokens+prompt) |
| `output/{T001,T002,...}.py` | S8 | Per-task scaffold Python scripts |
| `cron_schedule.json` | S8 | Cron schedule: task IDs, delay_seconds, dependencies, file paths |
| `tasks.json` | S8 | Task manifest: id, title, depends_on, layer, est_seconds, file, status |
| `tests.json` | S8 | Test manifest: task_id, test_code, valid flag |
| `export_artifact.json` | S8 | Full export: features, DAG, UI structure, theme, validation report, schedule |
| `session_{uuid}.jsonl` | S9 | Per-session telemetry (timestamp, step, input, output, latency_ms, tool_calls) |
| `training_data/stream.jsonl` | S9 | Continuous training stream (quality-filtered instruction/output pairs) |
| `output/finetune_dataset.jsonl` | S10 | Fine-tuning dataset in instruction/output JSONL format |
| `output/training_manifest.json` | S10 | Training manifest: session_id, dataset stats, torch/unsloth availability, readiness |
| `output/artifact_{session_id}.json` | S10 | Per-session artifact snapshot |
| `locks/{task_id}.lock` | S9 | Per-task file locks (created during cron_runner, removed after) |

---

## CLI Reference

| Command | Behavior |
|---------|----------|
| `python localAIv9.py "your prompt"` | Run full pipeline (S0–S12) with custom prompt |
| `python localAIv9.py` | Run pipeline with default prompt |
| `python localAIv9.py --ui` | Start NiceGUI dashboard on `127.0.0.1:8080` |
| `python localAIv9.py --dashboard` | Same as `--ui` |
| `python localAIv9.py --serve` | Same as `--ui` |
| `python localAIv9.py --execute` | Run cron runner (execute all scheduled tasks) |
| `python localAIv9.py --accelerate T001` | Mark task T001 as complete instantly |

---

## Dependencies

### Required (Gate-Checked by S0)
- `pydantic` — Data contracts and validation
- `httpx` — HTTP client for Ollama API and web search

### Optional (Auto-Detected)
- `nicegui` — Interactive web dashboard. Without it, dashboard runs in "mirror-only" mode.
- `torch` — Fine-tuning backend. Manifest reports availability.
- `unsloth` — Optimized fine-tuning. Manifest reports availability.

---

## Design Principles

1. **LLM is unreliable** — Every LLM call has caching, retry logic, recovery prompts, and deterministic fallbacks.
2. **Python enforces contracts** — Pydantic models normalize types, fill missing fields, and validate structure.
3. **Deterministic where possible** — S4–S8 are pure Python with no LLM calls.
4. **JSON-native** — All data flows through JSON/JSONL. All artifacts are JSON.
5. **Telemetry-first** — Every step is logged. Training data is collected continuously.
6. **No stubs, no placeholders** — Real LLM calls, real web searches, real file generation, real task execution.

---

## License

MIT
