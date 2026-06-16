# KiaOmni-Gaussian vs SnapKV vs Vanilla — Demo Results

**Model:** Qwen2.5-7B-Instruct · NF4 4-bit · ~4K token context · SDPA backend · Greedy decoding · Tesla T4  
**Budgets:** 98 · 128 · 256 · 512 · 1024 · 2048 · 3400 retained tokens · **N:** 10/10/10/3 samples per task

![Demo Results](demo_results.png)

---

## What the tasks measure

| Task | What it tests |
|------|--------------|
| **Single-needle** | Retrieve one 6-digit code hidden at a random depth in a filler haystack |
| **Multi-needle** | Retrieve 3 values (code, locker number, codename) planted at 20/50/80% depth |
| **Reasoning** | Trace a 4-hop variable assignment chain; a distractor chain is present |
| **Summary** | Cover all 8 planted numeric facts in a synthetic quarterly report |
| **Needle mean** | Mean accuracy over single + multi + reason (the primary signal) |

---

## Needle Mean Accuracy — Full Budget Sweep

| Budget | KiaOmni-Gaussian | SnapKV | Vanilla |
|--------|:---:|:---:|:---:|
| B=98   | 33.3% | 0.0% | 66.7% |
| B=128  | 33.3% | 0.0% | 66.7% |
| B=256  | 73.3% | 13.3% | 66.7% |
| **B=512**  | **80.0%** | 33.3% | 66.7% |
| B=1024 | 70.0% | 43.3% | 66.7% |
| B=2048 | 70.0% | 63.3% | 66.7% |
| B=3400 | 70.0% | 66.7% | 66.7% |

> KiaOmni **surpasses Vanilla at B=512** (80% vs 66.7%) using only 12.8% of the cache.  
> SnapKV needs **B=3400 (85% of cache)** to merely tie Vanilla.

## Per-Task Accuracy @ B=512

| Task | KiaOmni-Gaussian | SnapKV | Vanilla |
|------|:---:|:---:|:---:|
| Single-needle | **100%** | 80% | 100% |
| Multi-needle | **100%** | 20% | 100% |
| Reasoning | **40%** | 0% | 0% |
| Summary | **100%** | 100% | 100% |

## Efficiency @ B=512

| Method | tok/s | Peak VRAM | Gen-PPL |
|--------|:---:|:---:|:---:|
| Vanilla | 2.22 | 6.40 GB | 1.05 |
| **KiaOmni-Gaussian** | **2.89** | **3.15 GB** | **1.04** |
| SnapKV | 2.89 | 6.36 GB | 1.15 |

KiaOmni VRAM holds at **3.15 GB from B=98 through B=1024** — half of SnapKV at every budget.

---

> **Caveat:** Demo-scale results (small N, 4K context). Full evaluation: 61,681 samples · 4 models · [github.com/Aliw02/kiaomni](https://github.com/Aliw02/kiaomni)
