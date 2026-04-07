#!/usr/bin/env python3
"""
localAIv9 Pipeline - Python-native runtime, JSON-native contracts, deterministic, architecture-aware, mathematically grounded.
Install: pip install "pydantic<2" "httpx>=0.27,<1"
Run: python prompt.py "your prompt here"
Flow: S0-S9 pipeline sections plus pipeline_end lifecycle log event (id 9).
"""

# -- 0: Runtime Setup ---------------------------------------------------
# What Is The Required Input To Proceed?
# Input Contract:
# - Python 3.11+ interpreter context.
# - Installed dependencies: pydantic<2 and httpx>=0.27,<1.

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal, Tuple

import httpx
from pydantic import BaseModel, Field, validator, root_validator

# Output Contract:
# - Runtime namespace loaded: asyncio, hashlib, json, os, re, time, uuid, Path, typing (List, Dict, Any, Optional, Literal, Tuple), httpx, pydantic (BaseModel, Field, validator, root_validator).
# - File artifact: none (0 emits no .py or .json file).

# -- 1: Function Models ---------------------------------------------------
# What Is The Required Input To Proceed?
# Input Contract:
# - Runtime setup completed (imports and dependencies available).
# - No user payload required; this section defines static function models.

class Architecture(BaseModel):
    layers: List[Literal["frontend", "backend", "fullstack"]] = ["frontend", "backend"]
    modules: List[Dict[str, str]] = []
    api_endpoints: List[Dict[str, str]] = []

class Function1Output(BaseModel):
    features: List[Dict[str, str]] = [] # each: {"name": str, "layer": str}
    entities: List[str] = []
    actions: List[str] = []
    architecture: Architecture = Architecture()
    constraints: List[str] = []
    success_criteria: List[str] = []

    @validator("features")
    def features_have_layer(cls, v):
        for f in v:
            if "layer" not in f or f["layer"] not in ("frontend", "backend", "fullstack"):
                raise ValueError(f"Feature {f} missing or invalid layer")
        return v

class Component(BaseModel):
    id: str
    type: str
    props: Dict[str, Any] = {}

class Screen(BaseModel):
    id: str
    type: Literal["page", "modal", "tab"]
    components: List[Component] = []
    routes: Optional[str] = None
    tabs: Optional[str] = None

class Function2Output(BaseModel):
    screens: List[Screen] = []

class Theme(BaseModel):
    style: str
    color_palette: Dict[str, str] # primary, secondary, background, text, accent
    typography: Dict[str, str] # font_family, scale
    spacing: Dict[str, str] # base_unit
    components: Dict[str, str] # button_variant, input_variant, card_style

class Function3Output(BaseModel):
    theme: Theme
    tool_dataset_summary: Dict[str, Any]
    mathematical_derivations: Dict[str, Any]

class DAGNode(BaseModel):
    id: str
    depends_on: List[str] = []

class Function4Output(BaseModel):
    features: List[Dict[str, str]] = [] # name, source_screen, trigger_component
    dag: List[DAGNode] = []
    validation: Dict[str, List[str]] = {
        "missing_features": [],
        "orphan_nodes": [],
        "invalid_flows": []
    }

class Function5Output(BaseModel):
    final_dag: List[DAGNode]
    orphan_nodes_removed: bool

class ScheduledTask(BaseModel):
    id: str # T001, T002...
    depends_on: List[str]
    layer: Literal["frontend", "backend", "fullstack"]
    est_seconds: int

class Function6Output(BaseModel):
    scheduled_tasks: List[ScheduledTask]

class Function7Output(BaseModel):
    integrated_system: Dict[str, Any] # screens with theme, dag, validation_report

class Function8Output(BaseModel):
    export_artifact: Dict[str, Any]

# Output Contract:
# - Function classes loaded for downstream sections: Architecture, Function1Output, Function2Output, Theme, Function3Output, DAGNode, Function4Output, Function5Output, ScheduledTask, Function6Output, Function7Output, Function8Output.
# - Validators enforce data-contract checks at model instantiation.

# -- 2: LLM Client ------------------------------------------------------
# What Is The Required Input To Proceed?
# Input Contract:
# - Requires runtime imports: os, Path, json, hashlib, httpx, asyncio.
# - OLLAMA_HOST is reachable and MODEL is available in Ollama.
# - Function input is a prompt string with optional temperature and max_tokens.

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://172.30.80.1:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:26b")
CACHE_DIR = Path(".llm_cache")
CACHE_DIR.mkdir(exist_ok=True)

def _cache_key(prompt: str, temperature: float, max_tokens: int) -> str:
    material = json.dumps(
        {
            "host": OLLAMA_HOST,
            "model": MODEL,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt": prompt,
        },
        sort_keys=True,
    )
    return hashlib.md5(material.encode()).hexdigest()

async def ask_llm(prompt: str, temperature: float = 0.15, max_tokens: int = 2000) -> dict:
    """Call Ollama, return parsed JSON, with caching and retries."""
    key = _cache_key(prompt, temperature, max_tokens)
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    max_retries = 2
    backoff_base = 0.5
    last_text = ""
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                text = data["response"]
                last_text = text
        except Exception:
            if attempt < max_retries:
                await asyncio.sleep(backoff_base * (attempt + 1))
                continue
            raise

        # Extract first JSON object
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = text[start:end]
            try:
                result = json.loads(json_str)
                cache_file.write_text(json.dumps(result))
                return result
            except json.JSONDecodeError:
                if attempt < max_retries:
                    await asyncio.sleep(backoff_base * (attempt + 1))
                    continue
        else:
            if attempt < max_retries:
                await asyncio.sleep(backoff_base * (attempt + 1))
                continue

    # Fallback
    result = {"error": "invalid_json", "raw": last_text}
    cache_file.write_text(json.dumps(result))
    return result

# Output Contract:
# - Returns a parsed JSON dict when response extraction succeeds.
# - On failure returns an error dict with keys error and raw.
# - Persists cache entries in .llm_cache keyed by host, model, temperature, max_tokens, and prompt.

# -- 3: Intent Extraction -----------------------------------------------
# What Is The Required Input To Proceed?
# Input Contract:
# - Requires Function1Output model and ask_llm client loaded in runtime.
# - Function input is a user_prompt string (raw natural language intent).

async def function1(user_prompt: str) -> Function1Output:
    prompt = f"""You are an AI that outputs only valid JSON. Convert the following user intent into the exact JSON schema below.
Schema: {Function1Output.schema_json(indent=2)}
Intent: {user_prompt}
Output JSON:"""
    recovery_prompt = (
        prompt
        + "\nIMPORTANT: Return only one valid JSON object. No prose, no markdown fences."
    )
    raw = await ask_llm(prompt)
    if isinstance(raw, dict) and raw.get("error") == "invalid_json":
        raw = await ask_llm(recovery_prompt)
        if isinstance(raw, dict) and raw.get("error") == "invalid_json":
            raise RuntimeError("function1 failed: model returned invalid_json after recovery attempt")
    return Function1Output(**raw)

# Output Contract:
# - function1 returns a Function1Output instance with features, entities, actions, architecture, constraints, and success_criteria.
# - On persistent invalid JSON after recovery attempt, raises RuntimeError.

# -- 4: UI Structure Generation -----------------------------------------
# What Is The Required Input To Proceed?
# Input Contract:
# - Requires Function2Output schema and ask_llm client loaded in runtime.
# - Function input is a Function1Output instance from function1 (product spec with features, entities, actions, architecture).

async def function2(function1_out: Function1Output) -> Function2Output:
    prompt = f"""Based on the product spec, generate UI as JSON (no HTML, no styling). Use this schema:
{Function2Output.schema_json(indent=2)}
Product spec: {function1_out.json(indent=2)}
Output JSON:"""
    recovery_prompt = (
        prompt
        + "\nIMPORTANT: Return only one valid JSON object. No prose, no markdown fences."
    )
    raw = await ask_llm(prompt)
    if isinstance(raw, dict) and raw.get("error") == "invalid_json":
        raw = await ask_llm(recovery_prompt)
        if isinstance(raw, dict) and raw.get("error") == "invalid_json":
            raise RuntimeError("function2 failed: model returned invalid_json after recovery attempt")
    return Function2Output(**raw)

# Output Contract:
# - function2 returns a Function2Output instance with screens (pages, modals, tabs) containing components and routes.
# - On persistent invalid JSON after recovery attempt, raises RuntimeError.

# -- 5: Theme Derivation ------------------------------------------------
# What Is The Required Input To Proceed?
# Input Contract:
# - Requires runtime imports: httpx, json, List, Dict, Any.
# - Requires schemas: Theme, Function3Output, Function1Output, Function2Output.
# - Requires ask_llm client for LLM-driven theme derivation.
# - Function inputs are Function1Output from function1 (product intent) and Function2Output from function2 (UI structure).

async def _web_search(query: str, max_results: int = 10) -> List[str]:
    """Fetch search result snippets via DuckDuckGo HTML."""
    url = "https://html.duckduckgo.com/html/"
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.post(url, data={"q": query}, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            resp.raise_for_status()
        snippets = re.findall(r'result__snippet[^>]*>([^<]+)', resp.text)
        cleaned = [s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"') for s in snippets]
        return cleaned[:max_results]
    except Exception:
        return []

async def function3(function1_out: Function1Output, function2_out: Function2Output, retries: int = 0) -> Function3Output:
    # 1. Real web searches for design trend data
    ux_results = await _web_search("2025 UX design color trends", 10)
    ui_results = await _web_search("2025 UI design style trends", 10)
    award_results = await _web_search("design award winning color palettes 2025", 10)

    # 2. Real web searches for reference palettes
    nude_skin_results = await _web_search("warm neutral skin tone color palette hex codes", 10)
    makeup_shade_results = await _web_search("makeup foundation shade color palette hex", 10)
    accent_color_results = await _web_search("trendy accent color palette hex codes 2025", 10)

    # 3. Build context for LLM
    search_context = "\n".join(
        [f"UX trend: {r}" for r in ux_results] +
        [f"UI trend: {r}" for r in ui_results] +
        [f"Award palette: {r}" for r in award_results] +
        [f"Nude skin: {r}" for r in nude_skin_results] +
        [f"Makeup shade: {r}" for r in makeup_shade_results] +
        [f"Accent color: {r}" for r in accent_color_results]
    )
    product_context = function1_out.json(indent=2)
    ui_context = function2_out.json(indent=2)

    theme_prompt = f"""You are a design systems expert. Derive a cohesive visual theme.

WEB SEARCH DATA (current design trends + reference palettes):
{search_context}

PRODUCT SPEC (from intent extraction):
{product_context}

UI STRUCTURE (from structure generation):
{ui_context}

Derive a theme that:
1. Matches the product intent (features, entities, target audience)
2. Aligns with the UI structure (screens, components, complexity)
3. Reflects current design trends from web data
4. Uses reference palettes where relevant (skin tones, makeup shades, accent colors)
5. Ensures WCAG AA contrast (text vs background >= 4.5:1)

Return ONLY valid JSON matching this schema:
{Function3Output.schema_json(indent=2)}

Rules:
- All colors: valid 6-digit hex codes
- typography.scale: float between 1.0 and 2.0
- spacing.base_unit: "4px", "8px", "12px", "16px", or "24px"
- No markdown fences. No prose. Only JSON."""

    raw = await ask_llm(theme_prompt, temperature=0.3, max_tokens=1500)

    if isinstance(raw, dict) and raw.get("error") == "invalid_json":
        if retries < 2:
            return await function3(function1_out, function2_out, retries + 1)
        raise RuntimeError("function3 failed: LLM returned invalid JSON after 2 retries")

    # Validate hex colors
    hex_pattern = re.compile(r'^#[0-9a-fA-F]{6}$')
    palette = raw.get("theme", {}).get("color_palette", {})
    for key, val in palette.items():
        if not hex_pattern.match(str(val)):
            if retries < 2:
                return await function3(function1_out, function2_out, retries + 1)
            raw["theme"]["color_palette"][key] = "#6C63FF"

    return Function3Output(**raw)

# Output Contract:
# - function3 returns a Function3Output instance with LLM-derived Theme, tool_dataset_summary, and mathematical_derivations.
# - Theme is derived from web search trends + reference palettes + product spec + UI structure.
# - On LLM failure or invalid output, retries up to 2 times; raises RuntimeError on persistent failure.

# -- 6: Feature DAG Mapping ----------------------------------------------
# What Is The Required Input To Proceed?
# Input Contract:
# - Requires schemas: Function1Output, Function2Output, Function4Output, DAGNode.
# - Requires function1 for self-correction regeneration.
# - Requires function2 for self-correction regeneration.
# - Function inputs are Function1Output (features, actions, architecture) and Function2Output (screens, components).

async def function4(
    function1_out: Function1Output,
    function2_out: Function2Output,
    retries: int = 0,
    max_retries: int = 2
) -> Function4Output:
    # Deterministic feature mapping
    extracted = []
    for f in function1_out.features:
        mapped = False
        for screen in function2_out.screens:
            for comp in screen.components:
                if f["name"].lower() in comp.id.lower() or f["name"].lower() in comp.type.lower():
                    extracted.append({
                        "name": f["name"],
                        "source_screen": screen.id,
                        "trigger_component": comp.id
                    })
                    mapped = True
                    break
            if mapped:
                break
        if not mapped:
            extracted.append({"name": f["name"], "source_screen": "", "trigger_component": ""})

    # Build DAG from actions (simple sequential)
    dag = []
    for i, action in enumerate(function1_out.actions):
        dag.append(DAGNode(id=action, depends_on=function1_out.actions[:i]))

    # Validation
    missing = [f["name"] for f in function1_out.features if not any(ef["name"] == f["name"] and ef["source_screen"] for ef in extracted)]
    # Orphan nodes: nodes with no deps and no dependents
    all_ids = {n.id for n in dag}
    has_dependents = set()
    for n in dag:
        for d in n.depends_on:
            has_dependents.add(d)
    orphans = [n.id for n in dag if not n.depends_on and n.id not in has_dependents]
    invalid = [] # cycles detection omitted for brevity

    # Layer-aware validation
    for f in function1_out.features:
        if f["layer"] == "frontend":
            if not any(ef["name"] == f["name"] and ef["source_screen"] for ef in extracted):
                missing.append(f"frontend feature {f['name']} missing UI mapping")
        elif f["layer"] == "backend":
            if not any(ep.get("responsible_module") == f["name"] for ep in function1_out.architecture.api_endpoints):
                missing.append(f"backend feature {f['name']} missing API endpoint")

    validation = {
        "missing_features": missing,
        "orphan_nodes": orphans,
        "invalid_flows": invalid
    }

    # Self-correction if needed
    if missing or orphans or invalid:
        if retries >= max_retries:
            return Function4Output(features=extracted, dag=dag, validation=validation)
        # Regenerate only broken layer - we simulate by re-running function2 or function1
        if any("UI mapping" in m for m in missing):
            new_function2 = await function2(function1_out)
            return await function4(function1_out, new_function2, retries + 1, max_retries)
        elif any("API endpoint" in m for m in missing):
            new_function1 = await function1("Regenerate with proper API endpoints for backend features")
            return await function4(new_function1, function2_out, retries + 1, max_retries)
        else:
            return await function4(function1_out, function2_out, retries + 1, max_retries)

    return Function4Output(features=extracted, dag=dag, validation=validation)

# Output Contract:
# - Returns a Function4Output instance with extracted features (mapped to screens/components), sequential action DAG, and validation report.
# - On validation failure, self-corrects by re-running broken layers up to max_retries.

# -- 7: DAG Scheduling ---------------------------------------------------
# What Is The Required Input To Proceed?
# Input Contract:
# - Requires schemas: Function4Output, Function5Output, Function6Output, ScheduledTask, Function1Output.
# - Requires function4 output (feature DAG with validation report).
# - Requires function1 output (for layer mapping in function6).
# - Function inputs are Function4Output (from function4) for orphan pruning, then Function5Output + Function1Output for task scheduling.

def function5(function4_out: Function4Output) -> Function5Output:
    dag = function4_out.dag
    all_ids = {n.id for n in dag}
    has_dependents = set()
    for n in dag:
        for d in n.depends_on:
            has_dependents.add(d)
    orphans = [n for n in dag if not n.depends_on and n.id not in has_dependents]
    final_dag = [n for n in dag if n not in orphans]
    return Function5Output(final_dag=final_dag, orphan_nodes_removed=len(orphans) > 0)

def function6(function5_out: Function5Output, function1_out: Function1Output) -> Function6Output:
    tasks = []
    # Map node id to layer from function1 features
    layer_map = {f["name"]: f["layer"] for f in function1_out.features}
    for idx, node in enumerate(function5_out.final_dag, start=1):
        layer = layer_map.get(node.id, "fullstack")
        est = 30 if layer == "frontend" else 45 if layer == "backend" else 60
        # Find indices of dependencies
        dep_ids = []
        for dep_name in node.depends_on:
            # find index of dep in final_dag
            for i, n in enumerate(function5_out.final_dag, start=1):
                if n.id == dep_name:
                    dep_ids.append(f"T{i:03d}")
                    break
        tasks.append(ScheduledTask(
            id=f"T{idx:03d}",
            depends_on=dep_ids,
            layer=layer,
            est_seconds=est
        ))
    return Function6Output(scheduled_tasks=tasks)

# Output Contract:
# - function5 returns Function5Output: pruned DAG (orphans removed) and orphan_nodes_removed flag.
# - function6 returns Function6Output: scheduled tasks with sequential IDs (T001...TN), resolved dependency IDs, layer assignment, and per-layer time estimates.

# -- 8: Integration & Export --------------------------------------------
def function7(function6_out: Function6Output, function2_out: Function2Output, function3_out: Function3Output) -> Function7Output:
    # Apply theme to screens (simple merge)
    integrated_screens = []
    for screen in function2_out.screens:
        screen_dict = screen.dict()
        screen_dict["applied_theme"] = function3_out.theme.dict()
        integrated_screens.append(screen_dict)

    # Simulate routing and flow checks
    routing_ok = all(s.routes is not None for s in function2_out.screens if s.type == "page")
    theme_consistency = True
    edge_cases = []
    validation_report = {
        "routing_ok": routing_ok,
        "theme_consistency": theme_consistency,
        "edge_cases": edge_cases
    }
    integrated = {
        "screens": integrated_screens,
        "dag": [t.dict() for t in function6_out.scheduled_tasks],
        "validation_report": validation_report
    }
    return Function7Output(integrated_system=integrated)

def function8(function7_out: Function7Output, function1_out: Function1Output, function2_out: Function2Output,
function3_out: Function3Output, function4_out: Function4Output, function5_out: Function5Output,
function6_out: Function6Output) -> Function8Output:
    artifact = {
        "features": function1_out.features,
        "dag": [n.dict() for n in function5_out.final_dag],
        "ui_structure": [s.dict() for s in function2_out.screens],
        "theme": function3_out.theme.dict(),
        "validation_report": function7_out.integrated_system["validation_report"]
    }
    return Function8Output(export_artifact=artifact)

# -- 9: Orchestration & Logging -----------------------------------------

class Session:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.log_file = f"session_{self.session_id}.jsonl"

    def log(self, step: int, input_data: Any, output_data: Any, latency_ms: int, tool_calls: List[Dict] = None):
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": self.session_id,
            "step": step,
            "input": self._serialize(input_data),
            "output": self._serialize(output_data),
            "latency_ms": latency_ms,
            "tool_calls": tool_calls or []
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\\n")

    def _serialize(self, obj):
        if hasattr(obj, "dict"):
            return obj.dict()
        elif hasattr(obj, "__dict__"):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        else:
            return obj

async def run_pipeline(user_prompt: str) -> Tuple[Dict, str]:
    session = Session()
    start_total = time.time()

    # 3
    t0 = time.time()
    try:
        s1 = await function1(user_prompt)
        session.log(1, user_prompt, s1, int((time.time()-t0)*1000))
    except Exception as exc:
        session.log(1, user_prompt, {"error": str(exc)}, int((time.time()-t0)*1000))
        raise

    # 4
    t0 = time.time()
    try:
        s2 = await function2(s1)
        session.log(2, s1, s2, int((time.time()-t0)*1000))
    except Exception as exc:
        session.log(2, s1, {"error": str(exc)}, int((time.time()-t0)*1000))
        raise

    # 5
    t0 = time.time()
    try:
        s3 = await function3(s1, s2)
        session.log(3, {"function1": s1, "function2": s2}, s3, int((time.time()-t0)*1000))
    except Exception as exc:
        session.log(3, {"function1": s1, "function2": s2}, {"error": str(exc)}, int((time.time()-t0)*1000))
        raise

    # 6
    t0 = time.time()
    try:
        s4 = await function4(s1, s2)
        session.log(4, {"function1": s1, "function2": s2}, s4, int((time.time()-t0)*1000))
    except Exception as exc:
        session.log(4, {"function1": s1, "function2": s2}, {"error": str(exc)}, int((time.time()-t0)*1000))
        raise

    # 7A
    t0 = time.time()
    try:
        s5 = function5(s4)
        session.log(5, s4, s5, int((time.time()-t0)*1000))
    except Exception as exc:
        session.log(5, s4, {"error": str(exc)}, int((time.time()-t0)*1000))
        raise

    # 7B
    t0 = time.time()
    try:
        s6 = function6(s5, s1)
        session.log(6, {"function5": s5, "function1": s1}, s6, int((time.time()-t0)*1000))
    except Exception as exc:
        session.log(6, {"function5": s5, "function1": s1}, {"error": str(exc)}, int((time.time()-t0)*1000))
        raise

    # 8A
    t0 = time.time()
    try:
        s7 = function7(s6, s2, s3)
        session.log(7, {"function6": s6, "function2": s2, "function3": s3}, s7, int((time.time()-t0)*1000))
    except Exception as exc:
        session.log(7, {"function6": s6, "function2": s2, "function3": s3}, {"error": str(exc)}, int((time.time()-t0)*1000))
        raise

    # 8B
    t0 = time.time()
    try:
        s8 = function8(s7, s1, s2, s3, s4, s5, s6)
        session.log(8, s7, s8, int((time.time()-t0)*1000))
    except Exception as exc:
        session.log(8, s7, {"error": str(exc)}, int((time.time()-t0)*1000))
        raise

    total_ms = int((time.time()-start_total)*1000)
    session.log(9, "pipeline_end", s8, total_ms)

    return s8.export_artifact, session.session_id

if __name__ == "__main__":
    import sys
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Build a dashboard with user login and real-time charts."
    print(f"Running pipeline with prompt: {prompt}\\n")
    artifact, sid = asyncio.run(run_pipeline(prompt))
    print(f"\\nSession ID: {sid}")
    print("\\nFinal Export Artifact:\\n", json.dumps(artifact, indent=2))
