# MFBT DATASET — LIST 1 (HOST CEILING)
**Hardware:** 32GB VRAM
**Base unit:** 32 seed entries per category (dp0001–dp0256)
**Expansion:** 32× per category = **1,024 entries per category**
**Total records:** 8 × 1,024 = **8,192 JSONL entries**
**Loading order:** Frontier-first (hardest abstraction → application)

---

## SCHEMA (fixed across all categories)
```json
{
  "id": "dpXXXX",
  "text": "...",
  "source": "...",
  "url": "...",
  "timestamp": "YYYY-MM-DDTHH:MM:SSZ",
  "category": "...",
  "dependencies": []
}
```

---

## CATEGORY 1 — Scientific_Research
**Seed:** dp0001–dp0032 (32 entries)
**Expanded target:** dp0001–dp1024 (1,024 entries)
**Source datasets for expansion:**

| Rank | Source | HuggingFace ID | Field → `text` | Field → `url` | Records to pull |
|------|--------|---------------|----------------|---------------|-----------------|
| 1 | peS2o v3 | `allenai/peS2o` | `text` (abstract + body) | `source` URL | 600 |
| 2 | arXiv (RedPajama split) | `togethercomputer/RedPajama-Data-1T` | `text` | `meta.url` | 300 |
| 3 | PubMed abstracts | `ncbi/pubmed` | `MedlineCitation.Article.Abstract` | DOI link | 124 |

**category value:** `"scientific_research"`
**Frontload strategy:** Pull highest-cited papers first. peS2o provides `citation_count` field — sort descending.

**Verification:** https://github.com/allenai/peS2o | arXiv: https://arxiv.org

---

## CATEGORY 2 — Coding
**Seed:** dp0065–dp0096 (32 entries, Python-dominant)
**Expanded target:** 1,024 entries
**Source datasets for expansion:**

| Rank | Source | HuggingFace ID | Field → `text` | Records to pull |
|------|--------|---------------|----------------|-----------------|
| 1 | The Stack v2 smol | `bigcode/the-stack-v2-dedup` | `content` | 400 |
| 2 | StarCoderData (Python) | `bigcode/starcoderdata` | `content` | 300 |
| 3 | CodeSearchNet | `code_search_net` | `func_code_string` + `func_documentation_string` | 200 |
| 4 | StackOverflow (RedPajama) | `togethercomputer/RedPajama-Data-1T` | `text` | 124 |

**Language distribution for 1,024 entries:**
- Python: 300 | Rust: 120 | C/C++: 120 | JavaScript/TS: 100 | Go: 80 | Java: 80 | SQL: 80 | Bash: 80 | Other: 64

**category value:** `"coding"`
**Frontload strategy:** Pull by `max_stars_repo_count` descending — highest-starred repos carry most advanced patterns.

**Verification:** https://huggingface.co/datasets/bigcode/the-stack-v2 | Paper: arXiv:2402.19173

---

## CATEGORY 3 — Mathematics_Physics
**Seed:** dp0033–dp0064 (32 entries, formal proofs + theorems)
**Expanded target:** 1,024 entries
**Source datasets for expansion:**

| Rank | Source | HuggingFace ID | Field → `text` | Records to pull |
|------|--------|---------------|----------------|-----------------|
| 1 | Proof-Pile-2 (arXiv math+physics) | `EleutherAI/proof-pile-2` | `text` | 500 |
| 2 | OpenWebMath | `open-web-math/open-web-math` | `text` | 300 |
| 3 | MathPile (arXiv + textbooks) | `GAIR/MathPile_Commercial` | `text` | 224 |

**Subdomain distribution for 1,024 entries:**
- Formal proofs (Lean/Isabelle/Coq): 200
- arXiv math (analysis, algebra, topology): 300
- arXiv physics (hep-th, quant-ph, gr-qc): 200
- Applied math / numerical methods: 200
- Undergraduate textbook theorems: 124

**category value:** `"mathematics_physics"`
**Frontload strategy:** Proof-Pile-2 arxiv split sorted by `meta.timestamp` descending — most recent frontier math first.

**Verification:** Llemma: arXiv:2310.10631 | MathPile NeurIPS D&B 2024

---

## CATEGORY 4 — Language_Linguistics
**Seed:** dp0225–dp0256 (32 entries, multilingual + NLP concepts)
**Expanded target:** 1,024 entries
**Source datasets for expansion:**

| Rank | Source | HuggingFace ID | Field → `text` | Records to pull |
|------|--------|---------------|----------------|-----------------|
| 1 | Glot500 (rare languages) | `cis-lmu/Glot500` | `text` | 500 |
| 2 | Universal Dependencies | `universal_dependencies` | sentence + annotation | 300 |
| 3 | OPUS-100 (parallel) | `Helsinki-NLP/opus-100` | `translation` concat | 224 |

**Language distribution for 1,024 entries:**
- Low-resource languages (Glot500 tail): 400
- Syntactic annotation examples (UD): 300
- Parallel translation pairs (OPUS): 224
- Linguistic theory text: 100

**category value:** `"language_linguistics"`
**Frontload strategy:** Glot500 — pull least-represented language families first (maximizes structural diversity).

**Verification:** Glot500: arXiv:2305.12182 | UD: https://universaldependencies.org

---

## CATEGORY 5 — Human_Behavior_Social_Sciences
**Seed:** dp0193–dp0224 (32 entries, psychology + social dynamics)
**Expanded target:** 1,024 entries
**Source datasets for expansion:**

| Rank | Source | HuggingFace ID | Field → `text` | Records to pull |
|------|--------|---------------|----------------|-----------------|
| 1 | StackExchange (science/philosophy/cogsci) | RedPajama stackexchange split | `text` | 400 |
| 2 | OpenAssistant OASST2 | `OpenAssistant/oasst2` | `text` (assistant turns) | 300 |
| 3 | HH-RLHF | `Anthropic/hh-rlhf` | `chosen` field | 200 |
| 4 | Social Chemistry 101 | `allenai/social_chemistry_101` | `rot` + `situation` | 124 |

**category value:** `"human_behavior_social_sciences"`
**Frontload strategy:** StackExchange sorted by `score` descending — highest-voted answers = highest consensus human reasoning.

**Verification:** OASST2: https://huggingface.co/datasets/OpenAssistant/oasst2 | HH-RLHF: https://github.com/anthropics/hh-rlhf

---

## CATEGORY 6 — Engineering_Applied
**Seed:** dp0097–dp0128 (32 entries, patents + mechanical systems)
**Expanded target:** 1,024 entries
**Source datasets for expansion:**

| Rank | Source | HuggingFace ID | Field → `text` | Records to pull |
|------|--------|---------------|----------------|-----------------|
| 1 | HUPD (claims field) | `HUPD/hupd` | `claims` | 600 |
| 2 | BigPatent | `big_patent` | `description` | 300 |
| 3 | FreeLaw technical opinions | `pile-of-law/pile-of-law` | `text` (FreeLaw split) | 124 |

**IPC category distribution for 1,024 entries:**
- G06F (computing/electronics): 200
- H01/H02 (electrical): 150
- B60/B62 (transport/vehicles): 150
- C12/C22 (bio/materials): 150
- F01-F04 (mechanical/engines): 150
- A61 (medical devices): 100
- Other: 124

**category value:** `"engineering_applied"`
**Frontload strategy:** HUPD sorted by `patent_number` — higher patent numbers = more recent applied engineering.

**Verification:** HUPD: arXiv:2207.04043 | https://patentdataset.org

---

## CATEGORY 7 — Real_World_Execution
**Seed:** dp0129–dp0160 (32 entries, WikiHow/Instructables procedures)
**Expanded target:** 1,024 entries
**Source datasets for expansion:**

| Rank | Source | HuggingFace ID | Field → `text` | Records to pull |
|------|--------|---------------|----------------|-----------------|
| 1 | Super-NaturalInstructions | `Muennighoff/natural-instructions` | `definition` + `pos_examples` | 500 |
| 2 | WikiHow | `wikihow` | `text` (steps concat) | 300 |
| 3 | FLAN (task-specific subset) | `Muennighoff/flan` | `inputs` + `targets` | 224 |

**category value:** `"real_world_execution"`
**Frontload strategy:** Super-NI — pull tasks with highest `complexity` score first (multi-step, conditional logic tasks).

**Verification:** Super-NI: arXiv:2204.07705 | FLAN: arXiv:2109.01652

---

## CATEGORY 8 — Domain_Specific_Misc
**Seed:** dp0161–dp0192 (32 entries, legal + medical + regulatory)
**Expanded target:** 1,024 entries
**Source datasets for expansion:**

| Rank | Source | HuggingFace ID | Field → `text` | Records to pull |
|------|--------|---------------|----------------|-----------------|
| 1 | Pile of Law (FreeLaw + court opinions) | `pile-of-law/pile-of-law` | `text` | 400 |
| 2 | MultiLegal Pile (EN) | `joelniklaus/MultiLegalPile` | `text` | 300 |
| 3 | PhilPapers | `philosophersstone/philpapers` | `text` | 200 |
| 4 | FinGPT sentiment + news | `FinGPT/fingpt-sentiment-train` | `input` | 124 |

**Subdomain distribution for 1,024 entries:**
- Legal (US federal opinions): 300
- Legal (international/multilingual): 200
- Medical/regulatory (FDA, WHO, CDC): 200
- Philosophy: 200
- Finance: 124

**category value:** `"domain_specific_misc"`
**Frontload strategy:** Pile of Law — pull Supreme Court and Circuit Court opinions first (highest precedential density).

**Verification:** Pile of Law: arXiv:2207.09557 | MultiLegal: arXiv:2306.02069

---

## LIST 1 SUMMARY

| # | Category | Seed (existing) | Expanded | Primary Source |
|---|----------|----------------|----------|----------------|
| 1 | Scientific_Research | dp0001–dp0032 | 1,024 | allenai/peS2o |
| 2 | Coding | dp0065–dp0096 | 1,024 | bigcode/the-stack-v2-dedup |
| 3 | Mathematics_Physics | dp0033–dp0064 | 1,024 | EleutherAI/proof-pile-2 |
| 4 | Language_Linguistics | dp0225–dp0256 | 1,024 | cis-lmu/Glot500 |
| 5 | Human_Behavior | dp0193–dp0224 | 1,024 | RedPajama/stackexchange |
| 6 | Engineering_Applied | dp0097–dp0128 | 1,024 | HUPD/hupd |
| 7 | Real_World_Execution | dp0129–dp0160 | 1,024 | Muennighoff/natural-instructions |
| 8 | Domain_Specific_Misc | dp0161–dp0192 | 1,024 | pile-of-law/pile-of-law |
| | **TOTAL** | **256 entries** | **8,192 entries** | |

**ID continuation:** dp0257 → dp8448
**Disk footprint:** ~93.7GB source data → ~82MB final JSONL (text-only extracted)
**VRAM at training:** Peak ~15–18GB with batch_size=1, gradient_checkpointing=True

---

## EXTRACTION TEMPLATE (localAIv9 compatible)
```python
import json
from datasets import load_dataset

CATEGORY_MAP = {
    "scientific_research": ("allenai/peS2o", "text", "https://api.semanticscholar.org/"),
    "coding": ("bigcode/the-stack-v2-dedup", "content", "https://huggingface.co/datasets/bigcode/the-stack-v2"),
    "mathematics_physics": ("EleutherAI/proof-pile-2", "text", "https://huggingface.co/datasets/EleutherAI/proof-pile-2"),
    "language_linguistics": ("cis-lmu/Glot500", "text", "https://huggingface.co/datasets/cis-lmu/Glot500"),
    "human_behavior_social_sciences": ("OpenAssistant/oasst2", "text", "https://huggingface.co/datasets/OpenAssistant/oasst2"),
    "engineering_applied": ("HUPD/hupd", "claims", "https://patentdataset.org/"),
    "real_world_execution": ("Muennighoff/natural-instructions", "definition", "https://huggingface.co/datasets/Muennighoff/natural-instructions"),
    "domain_specific_misc": ("pile-of-law/pile-of-law", "text", "https://huggingface.co/datasets/pile-of-law/pile-of-law"),
}

def extract(category, hf_id, text_field, base_url, start_id, n=1024):
    ds = load_dataset(hf_id, split="train", streaming=True)
    records = []
    for i, row in enumerate(ds):
        if i >= n: break
        records.append({
            "id": f"dp{start_id + i:04d}",
            "text": row[text_field][:2048],  # cap at 2048 chars
            "source": hf_id,
            "url": base_url,
            "timestamp": row.get("timestamp", row.get("date", "2024-01-01T00:00:00Z")),
            "category": category,
            "dependencies": []
        })
    return records

# Run in frontier-first order
all_records = []
start = 257
for cat, (hf_id, field, url) in CATEGORY_MAP.items():
    batch = extract(cat, hf_id, field, url, start_id=start)
    all_records.extend(batch)
    start += 1024

with open("data/expanded_dataset_list1.jsonl", "w") as f:
    for r in all_records:
        f.write(json.dumps(r) + "\n")
```
