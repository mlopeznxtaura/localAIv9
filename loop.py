"""
loop.py — Modular Function Loop + Self-Evolving Transformer
One file. NiceGUI launches first. Terminal is background only.

Two data modes:
A — Ollama collection: loop runs, Ollama generates pairs, transformer trains at threshold
B — Load your data: provide JSONL, transformer trains immediately

PyTorch transformer trains on real data, saves checkpoint, loads on next run.
pip install nicegui ollama psutil torch
"""

import platform
import subprocess
import sys
import os
import time
import json
import hashlib
import threading
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR       = Path("training_data")
CANDIDATE_LOG  = DATA_DIR / "training_candidates.jsonl"
METRIC_LOG     = DATA_DIR / "metric_log.jsonl"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "transformer_checkpoint.pt"

DATA_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
SPAWN_PATH = Path("last_spawn.json")

# ── Tunable constants (base zero — user tunes these) ─────────────────────────

MAX_CYCLES           = 20
EVOLUTION_THRESHOLD  = 100   # pairs before transformer trains
OLLAMA_MODEL         = "qwen2.5-coder"

# ══════════════════════════════════════════════════════════════════════════════
# SPAWN POINT — stateless continuity across sessions
# Emitted at MAX_CYCLES or explicit STOP.
# Load last_spawn.json on next trigger to resume exactly.
# ══════════════════════════════════════════════════════════════════════════════

def emit_spawn(state: dict, host: dict, evolved: bool, checkpoint_exists: bool) -> dict:
    """
    Five keys: prompt, host, memory, ceiling, meta.
    Nothing else leaks forward. Saves to last_spawn.json.
    """
    spawn = {
        "prompt": state.get("prompt", ""),
        "host": {
            "os":      host.get("os"),
            "gpu":     host.get("gpu"),
            "cuda":    host.get("cuda"),
            "ollama":  host.get("ollama"),
            "pytorch": host.get("pytorch"),
        },
        "memory": {
            "data_path":       str(CANDIDATE_LOG),
            "metric_path":     str(METRIC_LOG),
            "checkpoint_path": str(CHECKPOINT_PATH) if checkpoint_exists else None,
            "total_pairs":     len(load_pairs()),
            "evolved":         evolved,
        },
        "ceiling": {
            "max_cycles":          MAX_CYCLES,
            "evolution_threshold": EVOLUTION_THRESHOLD,
        },
        "meta": {
            "iteration":    state.get("cycle_n", 1) - 1,
            "reliability":  state.get("reliability", 0.0),
            "last_quality": state.get("last_quality", 0.0),
            "last_route":   state.get("last_route", "none"),
            "spawned_at":   int(time.time()),
            "status":       "UNPLUGGED",
        },
    }
    try:
        with SPAWN_PATH.open("w", encoding="utf-8") as f:
            json.dump(spawn, f, indent=2)
        print(f" [SPAWN] saved → {SPAWN_PATH}")
    except Exception as e:
        print(f" [WARN] spawn save failed: {e}")
    return spawn

def load_spawn() -> dict:
    """Load last spawn point if it exists. Returns dict or None."""
    if not SPAWN_PATH.exists():
        return None
    try:
        with SPAWN_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════════
# HOST — read at launch, available to all cycles
# ══════════════════════════════════════════════════════════════════════════════

def read_host() -> dict:
    h = {
        "os": platform.system(), "arch": platform.machine(),
        "python": platform.python_version(), "cores": os.cpu_count(),
        "ram_gb": None, "gpu": None, "vram_gb": None,
        "ollama": False, "pkgs": [],
        "pytorch": False, "cuda": False, "cuda_device": None,
    }
    try:
        import psutil
        h["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            parts = r.stdout.strip().split(",")
            h["gpu"]     = parts[0].strip() if parts else None
            h["vram_gb"] = round(int(parts[1].strip())/1024,1) if len(parts)>1 else None
    except Exception:
        pass
    try:
        r = subprocess.run(["ollama","list"], capture_output=True, text=True, timeout=5)
        h["ollama"] = r.returncode == 0
    except Exception:
        pass
    try:
        import torch
        h["pytorch"] = True
        h["cuda"]    = torch.cuda.is_available()
        if h["cuda"]:
            h["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        r = subprocess.run([sys.executable,"-m","pip","list","--format=columns"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            h["pkgs"] = [l.split()[0] for l in r.stdout.splitlines()[2:] if l.strip()]
    except Exception:
        pass
    return h

# ══════════════════════════════════════════════════════════════════════════════
# ORIGIN POINT 1 — Training Data Logger
# ══════════════════════════════════════════════════════════════════════════════

def log_candidate(prompt: str, output: str, cycle: int, quality: float) -> dict:
    record = {
        "cycle":   cycle,
        "prompt":  prompt,
        "output":  output,
        "quality": quality,
        "ts":      int(time.time()),
        "hash":    hashlib.sha256((prompt+output).encode()).hexdigest()[:16],
        "label":   None,
    }
    with CANDIDATE_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record

def load_pairs(path: Path = CANDIDATE_LOG) -> list:
    if not path.exists():
        return []
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("prompt") and rec.get("output"):
                    pairs.append((rec["prompt"], rec["output"]))
            except Exception:
                continue
    return pairs

def load_user_data(path: Path) -> list:
    """
    Load user-provided JSONL.
    Accepts any of these field names:
      prompt/output, input/output, question/answer, user/assistant
    Skips malformed records silently.
    """
    pairs = []
    field_maps = [
        ("prompt","output"), ("input","output"),
        ("question","answer"), ("user","assistant"),
    ]
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    for pf, of in field_maps:
                        if pf in rec and of in rec:
                            pairs.append((str(rec[pf]), str(rec[of])))
                            break
                except Exception:
                    continue
    except Exception as e:
        print(f" [WARN] could not read user data: {e}")
    return pairs

# ══════════════════════════════════════════════════════════════════════════════
# ORIGIN POINT 2 — Metric Tracker
# ══════════════════════════════════════════════════════════════════════════════

ROUTE_SCORES = {"ollama":1.0,"transformer":1.0,"subprocess":0.6,"shell":0.4,"record":0.1}

def compute_quality(output: str, route: str, prompt: str) -> float:
    rs = ROUTE_SCORES.get(route, 0.1)
    ls = min(1.0, len(output.strip()) / max(1, len(prompt)*2))
    co = 0.0 if output.strip() == prompt.strip() else 1.0
    return round(rs*0.5 + ls*0.3 + co*0.2, 4)

def log_metric(cycle_n: int, reliability: float, quality: float, route: str):
    with METRIC_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "cycle_n": cycle_n, "reliability": reliability,
            "quality": quality, "route": route, "ts": int(time.time())
        }) + "\n")

# ══════════════════════════════════════════════════════════════════════════════
# PYTORCH TRANSFORMER — only imported/defined if PyTorch is available
# ══════════════════════════════════════════════════════════════════════════════

def build_and_train_transformer(pairs: list, emit, device_str: str = "cpu"):
    """
    Builds and trains a character-level MiniTransformer on pairs.
    Returns (model, vocab, inv_vocab) or None on failure.
    All torch imports are inside this function — safe if torch not installed.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset
    except ImportError:
        emit(" [FAIL] PyTorch not installed — run: pip install torch")
        return None, None, None

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    emit(f" [TRAIN] device={device} pairs={len(pairs)}")

    # Build character vocab
    all_text = "".join(p+o for p,o in pairs)
    chars = sorted(set(all_text))
    vocab = {ch: i+3 for i,ch in enumerate(chars)}
    vocab["<PAD>"] = 0; vocab["<EOS>"] = 1; vocab["<UNK>"] = 2
    inv_vocab = {v:k for k,v in vocab.items()}

    class TextDS(Dataset):
        def __init__(self, pairs, vocab, max_len=64):
            self.data = []
            for p,o in pairs:
                toks = [vocab.get(c, vocab["<UNK>"]) for c in (p+o)][:max_len]
                for i in range(1, len(toks)):
                    self.data.append((toks[:i], toks[i]))
        def __len__(self): return len(self.data)
        def __getitem__(self, idx):
            x,y = self.data[idx]
            return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    class MiniTransformer(nn.Module):
        def __init__(self, vsz, edim=128, nhead=4, nlayers=3, msl=64):
            super().__init__()
            self.tok_emb = nn.Embedding(vsz, edim, padding_idx=0)
            self.pos_emb = nn.Parameter(torch.zeros(1, msl, edim))
            self.layers  = nn.ModuleList([
                nn.TransformerDecoderLayer(edim, nhead, batch_first=True)
                for _ in range(nlayers)
            ])
            self.ln   = nn.LayerNorm(edim)
            self.head = nn.Linear(edim, vsz)
            self.msl  = msl

        def forward(self, x):
            sl  = x.size(1)
            emb = self.tok_emb(x) + self.pos_emb[:,:sl,:]
            msk = torch.triu(torch.full((sl,sl), float("-inf")), diagonal=1).to(x.device)
            for layer in self.layers:
                emb = layer(emb, emb, tgt_mask=msk)
            return self.head(self.ln(emb))

        def generate(self, tokens, max_new=50, temp=0.8):
            self.eval()
            gen = tokens
            with torch.no_grad():
                for _ in range(max_new):
                    logits = self.forward(gen)[0,-1,:] / temp
                    probs  = torch.nn.functional.softmax(logits, dim=-1)
                    nxt    = torch.multinomial(probs, 1)
                    gen    = torch.cat([gen, nxt.unsqueeze(0)], dim=1)
                    if nxt.item() == 1:
                        break
            return gen

    def collate(batch):
        xs, ys = zip(*batch)
        max_l  = max(x.size(0) for x in xs)
        xs_pad = torch.stack([
            torch.nn.functional.pad(x, (0, max_l - x.size(0))) for x in xs
        ])
        return xs_pad, torch.stack(ys)

    ds     = TextDS(pairs, vocab)
    loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate)
    model  = MiniTransformer(len(vocab)).to(device)
    opt    = optim.AdamW(model.parameters(), lr=1e-3)
    crit   = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    for epoch in range(10):
        total = 0
        for x,y in loader:
            x,y    = x.to(device), y.to(device)
            logits = model(x)
            loss   = crit(logits.view(-1, logits.size(-1)), y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        avg = total / max(1, len(loader))
        emit(f" [EPOCH {epoch+1}/10] loss={avg:.4f}")

    # Save checkpoint
    try:
        import torch
        torch.save({
            "model_state": model.state_dict(),
            "vocab":       vocab,
            "inv_vocab":   inv_vocab,
        }, CHECKPOINT_PATH)
        emit(f" [CHECKPOINT] saved → {CHECKPOINT_PATH}")
    except Exception as e:
        emit(f" [WARN] checkpoint save failed: {e}")

    return model, vocab, inv_vocab


def load_checkpoint(emit):
    """Load existing checkpoint if present. Returns (model, vocab, inv_vocab) or Nones."""
    if not CHECKPOINT_PATH.exists():
        return None, None, None
    try:
        import torch
        import torch.nn as nn

        ckpt      = torch.load(CHECKPOINT_PATH, map_location="cpu")
        vocab     = ckpt["vocab"]
        inv_vocab = ckpt["inv_vocab"]

        class MiniTransformer(nn.Module):
            def __init__(self, vsz, edim=128, nhead=4, nlayers=3, msl=64):
                super().__init__()
                self.tok_emb = nn.Embedding(vsz, edim, padding_idx=0)
                self.pos_emb = nn.Parameter(torch.zeros(1, msl, edim))
                self.layers  = nn.ModuleList([
                    nn.TransformerDecoderLayer(edim, nhead, batch_first=True)
                    for _ in range(nlayers)
                ])
                self.ln   = nn.LayerNorm(edim)
                self.head = nn.Linear(edim, vsz)
                self.msl  = msl
            def forward(self, x):
                sl  = x.size(1)
                emb = self.tok_emb(x) + self.pos_emb[:,:sl,:]
                msk = torch.triu(torch.full((sl,sl),float("-inf")),diagonal=1).to(x.device)
                for layer in self.layers:
                    emb = layer(emb, emb, tgt_mask=msk)
                return self.head(self.ln(emb))
            def generate(self, tokens, max_new=50, temp=0.8):
                self.eval()
                gen = tokens
                with torch.no_grad():
                    for _ in range(max_new):
                        logits = self.forward(gen)[0,-1,:] / temp
                        probs  = torch.nn.functional.softmax(logits, dim=-1)
                        nxt    = torch.multinomial(probs, 1)
                        gen    = torch.cat([gen, nxt.unsqueeze(0)], dim=1)
                        if nxt.item() == 1: break
                return gen

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = MiniTransformer(len(vocab)).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        emit(f" [CHECKPOINT] loaded from {CHECKPOINT_PATH} device={device}")
        return model, vocab, inv_vocab
    except Exception as e:
        emit(f" [WARN] checkpoint load failed: {e}")
        return None, None, None


def transformer_inference(model, vocab, inv_vocab, prompt: str) -> str:
    try:
        import torch
        device = next(model.parameters()).device
        tokens = [vocab.get(c, vocab["<UNK>"]) for c in prompt]
        inp    = torch.tensor([tokens], dtype=torch.long).to(device)
        out    = model.generate(inp, max_new=80, temp=0.8)
        ids    = out[0].tolist()[len(tokens):]
        return "".join(inv_vocab.get(i,"?") for i in ids if i not in (0,1)).strip()
    except Exception as e:
        return f"(transformer error: {e})"

# ══════════════════════════════════════════════════════════════════════════════
# ORIGIN POINT 3 — Fine-Tune Hook
# ══════════════════════════════════════════════════════════════════════════════

def finetune_hook(metric_history: list) -> dict:
    min_cycles = 10
    threshold  = 0.75
    if len(metric_history) < min_cycles:
        return {"triggered": False, "reason": "insufficient cycles"}
    avg = sum(m["quality"] for m in metric_history[-min_cycles:]) / min_cycles
    if avg < threshold:
        # ── WIRE REAL TRAINER HERE ──────────────────────────────────────
        # subprocess.run([sys.executable, "train.py", "--data", str(CANDIDATE_LOG)])
        # ────────────────────────────────────────────────────────────────
        return {"triggered": True, "reason": f"avg quality {avg:.4f} below {threshold}"}
    return {"triggered": False, "reason": f"avg quality {avg:.4f} above threshold"}

# ══════════════════════════════════════════════════════════════════════════════
# LOOP CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

class HarnessLoop:
    def __init__(self, host: dict):
        self.host          = host
        self.state         = None
        self.running       = False
        self.paused        = False
        self._inject       = None
        self._emit_fn      = None
        self._stat_fn      = None
        self._progress_fn  = None
        self._spawn_fn     = None
        self.transformer   = None
        self.vocab         = None
        self.inv_vocab     = None
        self.evolved       = False
        self.mode          = "ollama"   # "ollama" | "userdata"
        self.user_data_path = None

    def set_emitters(self, emit_fn, stat_fn, progress_fn=None, spawn_fn=None):
        self._emit_fn     = emit_fn
        self._stat_fn     = stat_fn
        self._progress_fn = progress_fn
        self._spawn_fn    = spawn_fn

    def emit(self, line, raw=False):
        if self._emit_fn: self._emit_fn(line, raw=raw)

    def stat(self, text, color="#5a5a72"):
        if self._stat_fn: self._stat_fn(text, color)

    def progress(self, n, total):
        if self._progress_fn: self._progress_fn(n, total)

    def trigger(self, initial_prompt: str, mode: str = "ollama", user_data_path: Path = None):
        if self.running: return
        self.mode           = mode
        self.user_data_path = user_data_path
        self.state = {
            "prompt":         initial_prompt,
            "history":        [],
            "metric_history": [],
            "cycle_n":        1,
            "reliability":    0.0,
        }
        self.running = True
        self.paused  = False
        threading.Thread(target=self._loop, daemon=True).start()

    def inject(self, p: str):
        self._inject = p

    def pause(self):
        self.paused = True; self.stat("PAUSED","#ffb800")

    def resume(self):
        self.paused = False; self.stat("RUNNING","#ff2d2d")

    def stop(self):
        self.running = False
        if self.state:
            spawn = emit_spawn(self.state, self.host, self.evolved, CHECKPOINT_PATH.exists())
            if self._spawn_fn:
                self._spawn_fn(spawn)
            self.stat("UNPLUGGED — spawn saved","#ffb800")

    def _maybe_evolve(self):
        """Check pair count and evolve if threshold reached."""
        if self.evolved:
            return
        pairs = load_pairs()
        self.progress(len(pairs), EVOLUTION_THRESHOLD)
        if len(pairs) >= EVOLUTION_THRESHOLD:
            self.emit(f"\n── EVOLUTION THRESHOLD REACHED ({len(pairs)} pairs) ──")
            self.emit("── TRAINING TRANSFORMER ──")
            device = "cuda" if self.host.get("cuda") else "cpu"
            m, v, iv = build_and_train_transformer(pairs, self.emit, device)
            if m is not None:
                self.transformer = m
                self.vocab       = v
                self.inv_vocab   = iv
                self.evolved     = True
                self.emit("── TRANSFORMER ACTIVE — inference engine switched ──")
                self.stat("TRANSFORMER ACTIVE","#00ff88")

    def _run_cycle(self) -> dict:
        state   = self.state
        cycle_n = state.get("cycle_n", 1)
        prompt  = state.get("prompt", "")
        history = state.get("history", [])

        self.emit(f"\n── CYCLE {cycle_n} ──────────────────────────")
        self.emit(f"INPUT {prompt[:120]}")

        output = ""; route = "none"; success = False

        # Transformer inference if evolved
        if self.evolved and self.transformer:
            try:
                output  = transformer_inference(self.transformer, self.vocab, self.inv_vocab, prompt)
                route   = "transformer"
                success = True
                self.emit(f"ROUTE transformer")
                self.emit(f"OUT {output[:200]}")
            except Exception as e:
                self.emit(f"ROUTE transformer failed: {e}")

        # Ollama
        if not success and self.host.get("ollama"):
            try:
                import ollama as _ollama
                self.emit("ROUTE ollama")
                route   = "ollama"
                context = "\n".join(
                    f"[{h['cycle_n']}] {h['output'][:120]}"
                    for h in history[-3:]
                )
                full   = f"{prompt}\n\nContext:\n{context}" if context else prompt
                tokens = []
                for chunk in _ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=[{"role":"user","content":full}],
                    stream=True
                ):
                    t = chunk["message"]["content"]
                    tokens.append(t)
                    self.emit(t, raw=True)
                output  = "".join(tokens)
                success = True
                self.emit("\n")
            except Exception as e:
                self.emit(f"ROUTE ollama failed: {e}")

        # Record only
        if not success:
            self.emit("ROUTE record-only")
            route   = "record"
            output  = prompt
            success = True

        # Metrics
        prior_r     = state.get("reliability", 0.0)
        cycle_ok    = 1.0 if route != "record" else 0.0
        reliability = round((prior_r*(cycle_n-1) + cycle_ok)/cycle_n, 4)
        quality     = compute_quality(output, route, prompt)

        log_candidate(prompt, output, cycle_n, quality)
        log_metric(cycle_n, reliability, quality, route)

        metric_history = state.get("metric_history",[]) + [{
            "cycle_n": cycle_n, "quality": quality, "reliability": reliability
        }]
        ft = finetune_hook(metric_history)
        if ft["triggered"]:
            self.emit(f"FINETUNE_HOOK TRIGGERED — {ft['reason']}")

        self.emit(f"QUALITY     {quality:.4f}")
        self.emit(f"RELIABILITY {reliability:.4f}")
        self.emit(f"RECORDED    cycle={cycle_n}")

        # Check evolution
        self._maybe_evolve()

        return {
            "prompt":         output.strip()[:500] if output.strip() else prompt,
            "history":        history + [{"cycle_n":cycle_n,"prompt":prompt,"output":output,"route":route}],
            "metric_history": metric_history,
            "cycle_n":        cycle_n + 1,
            "reliability":    reliability,
            "last_quality":   quality,
            "last_route":     route,
        }

    def _loop(self):
        self.stat("RUNNING","#ff2d2d")

        # Mode B — load user data first, train immediately
        if self.mode == "userdata" and self.user_data_path:
            self.emit(f"\n── LOADING USER DATA: {self.user_data_path} ──")
            pairs = load_user_data(self.user_data_path)
            self.emit(f"── {len(pairs)} valid pairs loaded ──")
            if pairs:
                for i,(p,o) in enumerate(pairs):
                    log_candidate(p, o, -(i+1), 1.0)
                self.emit("── TRAINING TRANSFORMER ON YOUR DATA ──")
                device = "cuda" if self.host.get("cuda") else "cpu"
                m,v,iv = build_and_train_transformer(pairs, self.emit, device)
                if m is not None:
                    self.transformer = m
                    self.vocab       = v
                    self.inv_vocab   = iv
                    self.evolved     = True
                    self.emit("── TRANSFORMER ACTIVE ──")
                    self.stat("TRANSFORMER ACTIVE","#00ff88")
            else:
                self.emit("── No valid pairs found — switching to Ollama collection ──")

        # Load existing checkpoint if not just trained
        if not self.evolved:
            m,v,iv = load_checkpoint(self.emit)
            if m is not None:
                self.transformer = m
                self.vocab       = v
                self.inv_vocab   = iv
                self.evolved     = True
                self.stat("CHECKPOINT LOADED — TRANSFORMER ACTIVE","#00ff88")

        # Main loop
        while self.running:
            if self._inject is not None:
                self.state["prompt"] = self._inject
                self._inject = None
                self.emit("\n── INJECTION — new prompt received ──\n")

            while self.paused and self.running:
                time.sleep(0.2)
            if not self.running:
                break

            # Stateless boundary — emit spawn point, unplug cleanly
            if self.state.get("cycle_n",1) > MAX_CYCLES:
                spawn = emit_spawn(self.state, self.host, self.evolved, CHECKPOINT_PATH.exists())
                self.emit(f"\n── MAX_CYCLES ({MAX_CYCLES}) REACHED — UNPLUGGED ──")
                self.emit(f"── spawn saved → {SPAWN_PATH} ──")
                self.emit(f"── total_pairs={spawn['memory']['total_pairs']} ──")
                self.emit(f"── evolved={spawn['memory']['evolved']} ──")
                self.emit("── trigger again to resume from spawn point ──")
                self.running = False
                self.stat(f"UNPLUGGED — spawn saved","#ffb800")
                if self._spawn_fn:
                    self._spawn_fn(spawn)
                break

            self.state = self._run_cycle()
            self.stat(
                f"CYCLE {self.state['cycle_n']-1} "
                f"quality={self.state['last_quality']:.4f} "
                f"reliability={self.state['reliability']:.4f} "
                f"route={self.state['last_route']} "
                f"{'[TRANSFORMER]' if self.evolved else f'[collecting {len(load_pairs())}/{EVOLUTION_THRESHOLD}]'}",
                "#00ff88"
            )
            time.sleep(1.5)

        if self.running:
            self.stat("LOOP ENDED","#5a5a72")

# ══════════════════════════════════════════════════════════════════════════════
# UI — NiceGUI launches first. Terminal is background only.
# ══════════════════════════════════════════════════════════════════════════════

from nicegui import ui

def build_ui(host: dict, loop: HarnessLoop):

    ui.add_head_html("""
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=Bebas+Neue&display=swap" rel="stylesheet">
    <style>
    *{box-sizing:border-box}
    body{background:#050508;font-family:'IBM Plex Mono',monospace;color:#c8c8d8;min-height:100vh}
    .title{font-family:'Bebas Neue',sans-serif;font-size:2.5rem;letter-spacing:.18em;color:#ff2d2d;text-shadow:0 0 40px rgba(255,45,45,.35)}
    .sub{font-size:.6rem;color:#3a3a52;letter-spacing:.12em}
    .host-bar{background:#0a0a12;border-left:3px solid #ff2d2d;padding:8px 14px;font-size:.67rem;color:#4a4a62;line-height:2}
    .origin-bar{background:#0a0a0f;border:1px solid #1a1a2e;border-left:3px solid #ffb800;padding:8px 14px;font-size:.63rem;color:#5a5a42;line-height:2}
    .mode-box{background:#0a0a12;border:1px solid #1a1a2e;padding:16px;border-radius:4px}
    .out{background:#060608;border:1px solid #111120;border-left:3px solid #1a1a2e;padding:16px;font-size:.72rem;color:#00ff88;min-height:320px;max-height:480px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;line-height:1.65;font-family:'IBM Plex Mono',monospace}
    .out.live{border-left-color:#ff2d2d}
    .stat{font-size:.66rem;letter-spacing:.08em;color:#3a3a52}
    .prog-wrap{background:#111120;height:6px;border-radius:3px;overflow:hidden}
    .prog-bar{height:6px;background:#ff2d2d;border-radius:3px;transition:width .4s}
    .prog-label{font-size:.62rem;color:#3a3a52;letter-spacing:.06em}
    .nicegui-input input,.nicegui-input textarea{background:#08080f!important;color:#e0e0f0!important;border:1px solid #1a1a2e!important;font-family:'IBM Plex Mono',monospace!important;font-size:.8rem!important}
    .btn-t{background:#ff2d2d!important;color:#050508!important;font-family:'Bebas Neue',sans-serif!important;font-size:1rem!important;letter-spacing:.1em!important;border-radius:2px!important}
    .btn-c{font-family:'IBM Plex Mono',monospace!important;font-size:.7rem!important;letter-spacing:.06em!important}
    </style>
    """)

    text_buf   = {"v": ""}
    mode_sel   = {"v": "ollama"}
    udata_path = {"v": None}

    host_str = (
        f"OS {host['os']} {host['arch']} // PY {host['python']} // "
        f"CORES {host['cores']} // RAM {host['ram_gb'] or '?'}GB // "
        f"GPU {host['gpu'] or 'none'} // VRAM {host['vram_gb'] or '?'}GB // "
        f"OLLAMA {'YES' if host['ollama'] else 'NO'} // "
        f"PYTORCH {'YES' if host['pytorch'] else 'NO'} // "
        f"CUDA {'YES — '+str(host['cuda_device']) if host['cuda'] else 'NO'}"
    )

    with ui.column().classes("w-full max-w-4xl mx-auto p-6 gap-4"):

        with ui.row().classes("items-end gap-4"):
            ui.html('<div class="title">META HARNESS</div>')
            ui.html('<div class="sub" style="padding-bottom:8px">MODULAR FUNCTION LOOP · SELF-EVOLVING TRANSFORMER · BASE ZERO</div>')

        ui.html(f'<div class="host-bar">{host_str}</div>')
        ui.html(
            '<div class="origin-bar">'
            f'ORIGIN 1 training_candidates.jsonl — Ollama-quality pairs per cycle<br>'
            f'ORIGIN 2 metric_log.jsonl — quality + reliability per cycle<br>'
            f'ORIGIN 3 finetune_hook() — wire trainer here · '
            f'EVOLUTION at {EVOLUTION_THRESHOLD} pairs · '
            f'STATELESS reset at MAX_CYCLES={MAX_CYCLES}'
            '</div>'
        )

        with ui.element("div").classes("mode-box"):
            ui.html('<div style="font-size:.65rem;color:#5a5a72;letter-spacing:.1em;margin-bottom:12px">SELECT DATA MODE</div>')
            with ui.row().classes("gap-6 items-start"):
                with ui.column().classes("gap-2"):
                    mode_a = ui.radio(["Ollama collection (first 100 cycles)"], value="Ollama collection (first 100 cycles)").props("color=red")
                    ui.html('<div style="font-size:.62rem;color:#3a3a52;max-width:260px">Ollama generates responses for first 100 cycles. Transformer trains on that data automatically.</div>')
                with ui.column().classes("gap-2"):
                    mode_b = ui.radio(["Load my own data (JSONL)"], value="Load my own data (JSONL)").props("color=red")
                    ui.html('<div style="font-size:.62rem;color:#3a3a52;max-width:260px">Provide your own JSONL. Transformer trains immediately. Fields: prompt/output, input/output, question/answer, user/assistant</div>')
                    data_path_input = ui.input(placeholder="/path/to/your/data.jsonl").classes("w-full").props("outlined dense")

        prompt_in = ui.textarea(placeholder="Initial intent. Speak once. Loop sustains itself.").classes("w-full").props("rows=3 outlined")

        prog_label = ui.html(f'<div class="prog-label">COLLECTING 0 / {EVOLUTION_THRESHOLD} pairs before transformer trains</div>')
        prog_wrap  = ui.html('<div class="prog-wrap"><div class="prog-bar" id="pb" style="width:0%"></div></div>')

        out_el  = ui.html('<div class="out"><pre>AWAITING TRIGGER...</pre></div>')
        stat_el = ui.html('<div class="stat">IDLE</div>')

        inject_in = ui.input(placeholder="Inject new prompt at next cycle boundary...").classes("w-full").props("outlined dense")

        with ui.row().classes("gap-3 items-center flex-wrap"):

            def on_trigger():
                p = (prompt_in.value or "").strip()
                if not p: return
                text_buf["v"] = ""
                if (data_path_input.value or "").strip():
                    sel_mode = "userdata"
                    udata    = Path(data_path_input.value.strip())
                else:
                    sel_mode = "ollama"
                    udata    = None
                loop.trigger(p, mode=sel_mode, user_data_path=udata)

            def on_inject():
                p = (inject_in.value or "").strip()
                if p:
                    loop.inject(p)
                    inject_in.set_value("")

            def emit(token, raw=False):
                if raw: text_buf["v"] += token
                else:   text_buf["v"] += "\n" + token
                safe = (text_buf["v"]
                        .replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
                out_el.set_content(f'<div class="out live"><pre>{safe}</pre></div>')

            def stat(text, color="#5a5a72"):
                stat_el.set_content(f'<div class="stat" style="color:{color}">{text}</div>')

            def progress(n, total):
                pct = min(100, int(n/total*100))
                prog_label.set_content(f'<div class="prog-label">COLLECTING {n} / {total} pairs before transformer trains</div>')
                prog_wrap.set_content(f'<div class="prog-wrap"><div class="prog-bar" style="width:{pct}%"></div></div>')

            def on_spawn(spawn: dict):
                pairs   = spawn.get("memory",{}).get("total_pairs", 0)
                evolved = spawn.get("memory",{}).get("evolved", False)
                itr     = spawn.get("meta",{}).get("iteration", 0)
                stat(f"UNPLUGGED iteration={itr} pairs={pairs} evolved={evolved} spawn → last_spawn.json","#ffb800")
                emit(f"\n── SPAWN POINT SAVED → last_spawn.json ──")
                emit(f"── iteration={itr} pairs={pairs} evolved={evolved} ──")
                emit(f"── trigger again to resume from this exact state ──")

            prior_spawn = load_spawn()
            if prior_spawn:
                prior_prompt = prior_spawn.get("prompt","")
                if prior_prompt:
                    prompt_in.set_value(prior_prompt[:200])
                stat(
                    f"PRIOR SPAWN FOUND iteration={prior_spawn.get('meta',{}).get('iteration',0)} "
                    f"pairs={prior_spawn.get('memory',{}).get('total_pairs',0)} "
                    f"evolved={prior_spawn.get('memory',{}).get('evolved',False)}",
                    "#ffb800"
                )

            loop.set_emitters(emit, stat, progress, spawn_fn=on_spawn)

            ui.button("TRIGGER",       on_click=on_trigger).classes("btn-t")
            ui.button("INJECT →",      on_click=on_inject).classes("btn-c").props("flat").style("color:#ffb800")
            ui.button("PAUSE/RESUME",  on_click=lambda: loop.resume() if loop.paused else loop.pause()).classes("btn-c").props("flat").style("color:#5a5a72")
            ui.button("STOP",          on_click=loop.stop).classes("btn-c").props("flat").style("color:#ff2d2d")
            ui.button("CLEAR",         on_click=lambda: (
                text_buf.update({"v":""}),
                out_el.set_content('<div class="out"><pre>CLEARED</pre></div>')
            )).classes("btn-c").props("flat").style("color:#3a3a52")

# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

host = read_host()
loop = HarnessLoop(host)
build_ui(host, loop)
ui.run(title="Meta Harness", port=7000, reload=False, dark=True)

# OUTPUT CONTRACT:
# training_candidates.jsonl — Ollama-quality pairs, one per cycle
# metric_log.jsonl          — quality + reliability per cycle
# transformer_checkpoint.pt — trained model, loads automatically next run
# Loop state per cycle: prompt, history, metric_history, cycle_n, reliability, quality
# Loop stops at MAX_CYCLES or STOP. Checkpoint and logs survive.
# Next trigger loads checkpoint automatically — no retraining needed.
