"""Generate curated CSVs, plots, provenance, and README for KiaOmni Qwen2.5-7B report."""

from pathlib import Path
import json, csv, shutil

SRC = Path(r"D:\MyFolder\ProgrammingWith-Python\Ai\A+\notebook\kv_cache_benchmark\033_full_comparison_results")
DST = Path(r"D:\MyFolder\ProgrammingWith-Python\Ai\A+\reports\qwen2.5-7b")
MAIN = Path(r"D:\MyFolder\ProgrammingWith-Python\Ai\A+\main-results\qwen2.5-7b")
DATA = DST / "data"
PLOTS = DST / "plots"

POLICY_MAP = {
    "FullContext": "FullContext",
    "H2O": "H2O",
    "RealSnapKV": "SnapKV",
    "SnapKV_Modified": "BlockSal",
    "Ada-SnapKV": "AdaSnapKV",
    "KiaOmni_\u03c38": "KiaOmni_\u03c38",
    "KiaOmni_Gaussian": "KiaOmni_Gaussian",
}
WHITELIST = set(POLICY_MAP.keys())

provenance: list[dict] = []
provenance_seq = [0]

def trace(source: str, table: str, cell: str | None = None, note: str | None = None):
    provenance_seq[0] += 1
    entry = {"id": provenance_seq[0], "source": source, "table": table}
    if cell:
        entry["cell"] = cell
    if note:
        entry["note"] = note
    provenance.append(entry)
    return entry["id"]

ENCD = dict(encoding="utf-8")

def filter_csv(src: Path, dst: Path):
    rows: list[list[str]] = []
    with open(src, newline="", **ENCD) as f:
        reader = csv.reader(f)
        h = next(reader)
        rows.append(h)
        for row in reader:
            if row[4] in WHITELIST:
                row[4] = POLICY_MAP[row[4]]
                rows.append(row)
    with open(dst, "w", newline="", **ENCD) as f:
        csv.writer(f).writerows(rows)

# ── 1. Filter speed_vram.csv ──
trace("speed_vram.csv", "speed_vram", "whitelist filtered")
filter_csv(SRC / "speed_vram.csv", DATA / "speed_vram.csv")

# ── 2. Filter speed_vram_all_contexts.csv ──
trace("speed_vram_all_contexts.csv", "speed_vram_all_contexts", "whitelist filtered")
src2 = SRC / "speed_vram_all_contexts.csv"
if src2.exists():
    filter_csv(src2, DATA / "speed_vram_all_contexts.csv")

# ── 3. Filter ruler_all_contexts_scored.csv ──
trace("ruler_all_contexts_scored.csv", "ruler_all_contexts_scored", "whitelist filtered")
src3 = SRC / "ruler_all_contexts_scored.csv"
if src3.exists():
    filter_csv(src3, DATA / "ruler_all_contexts_scored.csv")

# ── 4. Copy raw predictions ──
for fname in ["predictions_lb_scored.csv", "predictions_ruler_scored.csv"]:
    src = SRC / fname
    if src.exists():
        shutil.copy2(src, DATA / fname)
        trace(fname, fname, f"raw copy to data/")

# ── 5. Copy all source files to main-results ──
MAIN.mkdir(parents=True, exist_ok=True)
for f in SRC.iterdir():
    if f.is_file():
        shutil.copy2(f, MAIN / f.name)
trace("033_full_comparison_results/*", "main-results copy", "bulk archive")

# ── 6. Write provenance.json ──
PER_TASK = {
    "overall": "section-3-2 overall comparison table",
    "niah_sin": "section-3-2 per-task matrix row",
    "niah_mul": "section-3-2 per-task matrix row",
    "vt": "section-3-2 per-task matrix row",
    "narrativeqa": "section-3-2 per-task matrix row",
    "qasper": "section-3-2 per-task matrix row",
    "multifieldqa_en": "section-3-2 per-task matrix row",
    "hotpotqa": "section-3-2 per-task matrix row",
    "2wikimqa": "section-3-2 per-task matrix row",
    "musique": "section-3-2 per-task matrix row",
    "gov_report": "section-3-2 per-task matrix row",
    "qmsum": "section-3-2 per-task matrix row",
}
for col, cell_ref in PER_TASK.items():
    trace("final_analysis_report.md", "final_scores.csv", cell_ref, f"column={col}")

with open(DST / "provenance.json", "w", **ENCD) as f:
    json.dump(provenance, f, indent=2)

print(f"Provenance: {len(provenance)} entries written.")
print("All CSVs, copies done.")
