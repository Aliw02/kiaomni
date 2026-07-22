# Project: AI Research Workspace
**Owner:** Ali  
**Stack:** Python 3.13 · PyTorch · Transformers (HuggingFace) · NumPy · Pandas  
**Focus:** LLM KV-Cache research, inference optimization, benchmarking

---

## 📁 Key Directories
| Path | Purpose |
|------|---------|
| `d:/MyFolder/ProgrammingWith-Python/Ai/A+` | Main research workspace |
| `d:/MyFolder/ProgrammingWith-Python/Ai/A+/.Codex/skills/` | Custom Codex skills |

---

## 🧠 Project Context
- **Research:** KiaCache / KiaCachePlusR2 — a paged attention KV-cache eviction strategy
- **Models tested:** TinyLlama-1.1B-Chat, larger LLMs
- **Benchmarks:** LongBench V2, Needle-in-Haystack, Passkey Retrieval, PPL
- **Training:** QLoRA + Knowledge Distillation approach
- **Eviction policies:** Mean, Max, Top3 block scoring

## 🔒 Production Algorithm Rules (MANDATORY)
- **Zero Hallucination:** You are treating this as a strictly scientific, production-ready algorithm. Never guess the mathematical logic or tensor dimensions of KiaCache or KiaCachePlusR2.
- **Read the Source of Truth:** Before proposing any architectural or algorithmic changes, you MUST use your tools to read `ALGORITHM_STORY.md` (or the equivalent core documentation) to refresh your exact memory of the project's evolution.
- **Production Standard:** All code must be production-ready: rigorously typed (`typing`), memory-safe (avoiding memory leaks during long-context generation), and fully verified via metrics before you claim it works.

---

## ⚙️ Coding Conventions
- **Language:** Python 3.13
- **Style:** PEP 8, type hints always, docstrings on every public function
- **Testing:** pytest, test files as `test_*.py`
- **Git workflow:** feature branches, descriptive commit messages

---

## 🛠️ Common Commands
```bash
# Run benchmarks
python eval.py --model tinyllama --benchmark longbench

# Run tests
pytest tests/ -v

# Install deps
pip install -r requirements.txt

# Git log
git log --oneline -10
```

---

## 🚀 Installed MCP Servers
| Name | Purpose |
|------|---------|
| `filesystem` | Read/write files across `d:/MyFolder` |
| `memory` | Persist key facts across sessions (knowledge graph) |
| `fetch` | Fetch and read URLs / web pages |
| `sequential-thinking` | Break complex problems into structured reasoning steps |
| `git` | Git operations (log, diff, commit, status) |
| `context7` | Up-to-date library docs (HuggingFace, PyTorch, etc.) |

---

## 📋 Available Skills (type `/skill-name`)
| Skill | Trigger | What it does |
|-------|---------|--------------|
| `code-review` | `/code-review` | Deep code review with security + perf checks |
| `debug` | `/debug` | Systematic bug diagnosis |
| `research` | `/research` | Web research + summarize findings |
| `git-workflow` | `/git-workflow` | Commit, branch, PR workflow |

---

## 🔑 API Keys Needed (optional)
- **GitHub MCP:** Set `GITHUB_TOKEN` env var for private repos
- **Brave Search MCP:** Set `BRAVE_API_KEY` env var (free tier: 2000 req/month)

---

## 💡 Tips
- Use `/clear` when switching between unrelated tasks
- Use `/compact` before context gets too large
- Use `/plan` before large refactors
- Memory MCP persists facts: ask Codex to "remember X" and it survives sessions
