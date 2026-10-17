# Research-only auto-activated True-KV V2

This branch is **only** for the Qwen2.5-7B reproducible True-KV GPU ablation.
It is not the licensed SDK release, the Ahmed Zaki evaluation bundle, or a
production distribution.

## Access policy

The original SDK and its license hash are unchanged on their original branches.
The research branch allows **only an explicit** environment opt-in:
`KIAOMNI_TRUEKV_RESEARCH_ONLY=1`. The companion Kaggle notebook sets this flag
for its own Python import preflight and controlled ablation subprocess. No
Kaggle Secret, private license, random token, or license value is used.

The public `KiaOmniModel` wrapper continues to enforce its own SDK evaluation
license; the research runner imports internal modules directly, not that wrapper.
**Do not merge the root package research bypass into any protected/release
branch**. This opt-in is intentionally not a secure DRM mechanism.

## Run gates

1. Clone the exact research branch commit to a new directory.
2. Check Python imports and offline protocol tests, confirming the evaluation
   flag is set.
3. Run only smoke and mandatory preflight (1K equivalence gate).
4. If the preflight fails, inspect `preflight.json` and stop before pilot.
5. Only after smoke passes consider 4K pilot and 8K final on identical GPUs.

Keep GPU results separate from the old licensed/experimental output directory.
Neither Python compilation nor this repo access check constitutes a real GPU pass.
