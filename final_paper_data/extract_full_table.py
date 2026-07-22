"""
extract_full_table.py  v2
Produces: full_results_table.csv + full_results_table.md
4 models x 4 budgets x tasks x 7 policies -> CORRECT%
Verdict column: judge_label  (values: CORRECT / HALLUCINATED / REFUSED / NOISE)
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent

MODELS = {
    "Qwen2.5-7B":   BASE / "033_full_comparison_results" / "llm_judge_results.csv",
    "Mistral-7B":   BASE / "034_mistral_results" / "data" / "llm_judge_results.csv",
    "Falcon3-7B":   BASE / "037_falcon3_results" / "llm_judge_results.csv",
    "BioMistral-7B":BASE / "038_biomistral_results"      / "llm_judge_results.csv",
}

BUDGETS = [98, 128, 256, 512]

POLICY_NORM = {
    "fullcontext":            "FullContext",
    "full_context":           "FullContext",
    "kiaomni_gaussian":       "KiaOmni_Gaussian",
    "kiaomni_sigma8":         "KiaOmni_σ8",
    "kiaomni_s8":             "KiaOmni_σ8",
    "kiaomni_σ8":             "KiaOmni_σ8",
    "blocksalience":          "BlockSal",
    "blocksal":               "BlockSal",
    "snapkv_modified":        "BlockSal",
    "ada-snapkv":             "Ada-SnapKV",
    "ada_snapkv":             "Ada-SnapKV",
    "adasnapkv":              "Ada-SnapKV",
    "h2o":                    "H2O",
    "realsnapkv":             "RealSnapKV",
    "real_snapkv":            "RealSnapKV",
    # non-paper policies (present in raw data, excluded from paper table)
    "kiaomni_adaptive":       "_skip",
    "kiaomni_anchorexp":      "_skip",
    "kiaomni_quest":          "_skip",
    "kiaomni_ratioadaptive":  "_skip",
    "kiaomni_scissorhands":   "_skip",
}

# The 7 policies we want in the paper table
PAPER_POLICIES = ["FullContext", "KiaOmni_Gaussian", "KiaOmni_σ8",
                  "BlockSal", "Ada-SnapKV", "H2O", "RealSnapKV"]

rows = []

for model_name, csv_path in MODELS.items():
    print(f"\n=== {model_name} ===")
    if not csv_path.exists():
        print(f"  MISSING: {csv_path}")
        continue
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  {len(df):,} rows")

    # normalise policy
    df["pol"] = df["policy"].str.strip().str.lower().map(POLICY_NORM).fillna(df["policy"].str.strip())
    df = df[df["pol"] != "_skip"]

    # normalise budget  (Falcon3 uses 96, map→98 for table alignment)
    df["bud"] = pd.to_numeric(df["budget"], errors="coerce").astype("Int64")
    df.loc[df["bud"] == 96, "bud"] = 98  # Falcon3 lowest budget → align to B=98 slot

    # verdict
    df["correct"] = df["judge_label"].str.upper().str.strip() == "CORRECT"

    tasks = sorted(df["task"].str.strip().str.lower().unique())
    print(f"  Tasks ({len(tasks)}): {tasks}")
    print(f"  Policies: {sorted(df['pol'].unique())}")

    for task in tasks:
        for budget in BUDGETS:
            row = {"Model": model_name, "Task": task, "Budget": budget}
            mask_tb = (df["task"].str.strip().str.lower() == task) & (df["bud"] == budget)
            sub_tb = df[mask_tb]
            for pol in PAPER_POLICIES:
                sub_p = sub_tb[sub_tb["pol"] == pol]
                if len(sub_p) == 0:
                    row[pol] = "—"
                else:
                    row[pol] = f"{sub_p['correct'].mean()*100:.1f}"
            rows.append(row)

result = pd.DataFrame(rows)
out_csv = BASE / "full_results_table.csv"
result.to_csv(out_csv, index=False)
print(f"\nSaved CSV ({len(result)} rows): {out_csv}")

# ── Markdown ────────────────────────────────────────────────────────────────
md = ["# Full Results: CORRECT% by Model × Budget × Task × Policy\n"]
md.append("*Raw source: `llm_judge_results.csv` per experiment. Verdict: `judge_label == CORRECT`.*\n")
md.append("*Falcon3 B=96 shown under B=98 column for alignment.*\n")

for model_name in MODELS:
    md.append(f"\n## {model_name}\n")
    sub_m = result[result["Model"] == model_name]
    tasks = sorted(sub_m["Task"].unique())

    for budget in BUDGETS:
        md.append(f"\n### B={budget}\n")
        sub_b = sub_m[sub_m["Budget"] == budget]
        header = "| Task | " + " | ".join(PAPER_POLICIES) + " |"
        sep    = "|------|" + "|------" * len(PAPER_POLICIES) + "|"
        md.append(header); md.append(sep)
        for task in tasks:
            r = sub_b[sub_b["Task"] == task]
            if r.empty:
                continue
            vals = [str(r.iloc[0].get(p, "—")) for p in PAPER_POLICIES]
            md.append(f"| {task} | " + " | ".join(vals) + " |")
        # Macro avg
        def avg_col(col):
            nums = [float(v) for v in sub_b[col] if isinstance(v, str) and v not in ("—",)]
            if not nums: return "—"
            return f"**{sum(nums)/len(nums):.1f}**"
        avgs = [avg_col(p) for p in PAPER_POLICIES]
        md.append("| **MACRO AVG** | " + " | ".join(avgs) + " |")

out_md = BASE / "full_results_table.md"
out_md.write_text("\n".join(md), encoding="utf-8")
print(f"Saved MD: {out_md}")
print("\nDone.")
