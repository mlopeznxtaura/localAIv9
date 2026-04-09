"""
BRIDGE.py — Connects localAIv9 S9 telemetry to loop.py Mode B

Watches training_data/stream.jsonl (written by localAIv9 S9 every pipeline run)
Filters records — excludes errors, null outputs, incomplete pairs
Converts valid records to loop.py Mode B compatible JSONL (fields: prompt, output)
Writes to training_data/bridge_output.jsonl
Triggers loop.py Mode B with that file as input

Callable two ways:
- Standalone: python BRIDGE.py
- Importable: from BRIDGE import run_bridge (called directly from localAIv9 S9)

No new dependencies. Matches exact field names from both files.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Tuple

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR          = Path("training_data")
STREAM_FILE       = DATA_DIR / "stream.jsonl"
BRIDGE_OUTPUT     = DATA_DIR / "bridge_output.jsonl"
LOOP_PY_PATH      = Path("loop.py")

DATA_DIR.mkdir(exist_ok=True)


def _is_valid_record(record: dict) -> bool:
    """
    Validate a stream.jsonl record for bridge conversion.
    
    Excludes:
    - Records with 'error' in output (teacher_raw)
    - Records with null/None outputs
    - Records missing instruction or teacher_raw
    - Records where instruction or teacher_raw are empty strings
    """
    # Check for error in output
    teacher_raw = record.get("teacher_raw")
    if teacher_raw is None:
        return False
    
    # Parse teacher_raw to check for errors
    try:
        output_obj = json.loads(teacher_raw)
        if isinstance(output_obj, dict) and "error" in output_obj:
            return False
    except (json.JSONDecodeError, TypeError):
        # If we can't parse it, treat as invalid
        return False
    
    # Check instruction exists and is not empty
    instruction = record.get("instruction")
    if instruction is None or instruction == "":
        return False
    
    # Check teacher_raw is not empty or null representation
    if teacher_raw == "" or teacher_raw == "null":
        return False
    
    return True


def _extract_prompt_output(record: dict) -> Optional[Tuple[str, str]]:
    """
    Extract prompt and output from a stream.jsonl record.
    
    Priority for prompt:
    1. user_prompt (if present and non-empty string)
    2. instruction (parsed from JSON)
    
    Output is always teacher_raw (parsed from JSON).
    
    Returns (prompt, output) tuple or None if extraction fails.
    """
    # Extract output from teacher_raw
    teacher_raw = record.get("teacher_raw")
    try:
        output_obj = json.loads(teacher_raw)
        # Convert to string representation for consistency
        if isinstance(output_obj, (dict, list)):
            output = json.dumps(output_obj, ensure_ascii=False)
        else:
            output = str(output_obj)
    except (json.JSONDecodeError, TypeError):
        output = teacher_raw
    
    if not output or output.strip() == "":
        return None
    
    # Extract prompt - prefer user_prompt if available
    user_prompt = record.get("user_prompt")
    if user_prompt and isinstance(user_prompt, str) and user_prompt.strip():
        prompt = user_prompt
    else:
        # Fall back to instruction
        instruction = record.get("instruction")
        try:
            instruction_obj = json.loads(instruction)
            if isinstance(instruction_obj, (dict, list)):
                prompt = json.dumps(instruction_obj, ensure_ascii=False)
            else:
                prompt = str(instruction_obj)
        except (json.JSONDecodeError, TypeError):
            prompt = instruction if instruction else ""
    
    if not prompt or prompt.strip() == "":
        return None
    
    return (prompt, output)


def filter_and_convert_stream(input_path: Path, output_path: Path) -> int:
    """
    Read stream.jsonl, filter invalid records, convert to Mode B format.
    
    Args:
        input_path: Path to stream.jsonl
        output_path: Path to write bridge_output.jsonl
        
    Returns:
        Number of valid records written
    """
    if not input_path.exists():
        print(f" [BRIDGE] Input file not found: {input_path}")
        return 0
    
    valid_count = 0
    
    with open(output_path, "w", encoding="utf-8") as out_f:
        with open(input_path, "r", encoding="utf-8") as in_f:
            for line in in_f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # Filter invalid records
                if not _is_valid_record(record):
                    continue
                
                # Extract prompt/output pair
                result = _extract_prompt_output(record)
                if result is None:
                    continue
                
                prompt, output = result
                
                # Write Mode B compatible format
                mode_b_record = {
                    "prompt": prompt,
                    "output": output
                }
                out_f.write(json.dumps(mode_b_record, ensure_ascii=False) + "\n")
                valid_count += 1
    
    return valid_count


def trigger_loop_mode_b(data_path: Path) -> bool:
    """
    Trigger loop.py in Mode B with the given data file.
    
    Imports loop.py functions directly and calls load_user_data to train.
    Avoids triggering NiceGUI by patching sys.modules before import.
    
    Args:
        data_path: Path to bridge_output.jsonl
        
    Returns:
        True if triggered successfully, False otherwise
    """
    if not LOOP_PY_PATH.exists():
        print(f" [BRIDGE] loop.py not found at {LOOP_PY_PATH}")
        return False
    
    if not data_path.exists():
        print(f" [BRIDGE] Data file not found: {data_path}")
        return False
    
    try:
        # Prevent nicegui from launching UI when importing loop.py
        # We temporarily mock ui module to be a no-op for all attributes
        import types
        
        # Create a comprehensive mock nicegui.ui module that returns self-chainable objects
        class MockElement:
            def __getattr__(self, name):
                return lambda *args, **kwargs: self
            def classes(self, *args, **kwargs):
                return self
            def props(self, *args, **kwargs):
                return self
            def style(self, *args, **kwargs):
                return self
            def on(self, *args, **kwargs):
                return self
        
        class MockUI:
            def __getattr__(self, name):
                return lambda *args, **kwargs: MockElement()
        
        mock_ui = MockUI()
        sys.modules['nicegui'] = types.ModuleType('nicegui')
        sys.modules['nicegui.ui'] = mock_ui
        sys.modules['nicegui'].ui = mock_ui
        
        # Import loop.py functions directly
        sys.path.insert(0, str(Path(".").resolve()))
        from loop import load_user_data, log_candidate, build_and_train_transformer, read_host
        
        print(f" [BRIDGE] Loading data from {data_path}...")
        pairs = load_user_data(data_path)
        
        if not pairs:
            print(f" [BRIDGE] No valid pairs found in {data_path}")
            return False
        
        print(f" [BRIDGE] Loaded {len(pairs)} valid pairs")
        
        # Log all pairs to training_candidates.jsonl (same as loop.py Mode B)
        for i, (p, o) in enumerate(pairs):
            log_candidate(p, o, -(i+1), 1.0)
        
        print(f" [BRIDGE] Logged {len(pairs)} pairs to training_candidates.jsonl")
        
        # Train transformer on loaded data
        host = read_host()
        device = "cuda" if host.get("cuda") else "cpu"
        
        def emit(msg):
            print(f" [LOOP] {msg}")
        
        print(f" [BRIDGE] Training transformer on device={device}...")
        model, vocab, inv_vocab = build_and_train_transformer(pairs, emit, device)
        
        if model is not None:
            print(f" [BRIDGE] Transformer trained successfully")
            return True
        else:
            print(f" [BRIDGE] Transformer training failed (PyTorch may not be installed)")
            return False
            
    except Exception as e:
        print(f" [BRIDGE] Failed to trigger loop.py functions: {e}")
        return False


def run_bridge(trigger_loop: bool = True) -> Tuple[int, bool]:
    """
    Main bridge function - filters stream.jsonl and optionally triggers loop.py.
    
    Args:
        trigger_loop: If True, triggers loop.py Mode B after conversion
        
    Returns:
        Tuple of (valid_record_count, loop_triggered_successfully)
    """
    print(f" [BRIDGE] Starting bridge process...")
    print(f" [BRIDGE] Input:  {STREAM_FILE}")
    print(f" [BRIDGE] Output: {BRIDGE_OUTPUT}")
    
    # Filter and convert
    valid_count = filter_and_convert_stream(STREAM_FILE, BRIDGE_OUTPUT)
    print(f" [BRIDGE] Converted {valid_count} valid records")
    
    if valid_count == 0:
        print(f" [BRIDGE] No valid records to process")
        return (0, False)
    
    # Optionally trigger loop.py
    loop_triggered = False
    if trigger_loop:
        loop_triggered = trigger_loop_mode_b(BRIDGE_OUTPUT)
    
    return (valid_count, loop_triggered)


if __name__ == "__main__":
    # Standalone execution
    valid_count, triggered = run_bridge(trigger_loop=True)
    
    if valid_count > 0:
        print(f" [BRIDGE] Complete: {valid_count} records → {BRIDGE_OUTPUT}")
        if triggered:
            print(f" [BRIDGE] loop.py Mode B triggered successfully")
        else:
            print(f" [BRIDGE] loop.py was not triggered")
    else:
        print(f" [BRIDGE] No valid records found in {STREAM_FILE}")
        sys.exit(0)
