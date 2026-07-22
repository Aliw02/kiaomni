"""Integration test: verify the chat-template saliency mask produces
correct Q1-Q9 answers on the Starship article.

Requires a GPU and a loaded Qwen2.5-7B-Instruct model. Skipped when
CUDA is not available — runs only on the Modal deployment.

Usage:
    modal run kiaomni_chat/modal_app.py::test_q1_q9_acceptance
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent.parent


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_select_keep_with_chat_template_mask_produces_user_content_dominated_mask():
    """After masking, the keep_indices must contain predominantly user
    content positions (not chat-template structure tokens)."""
    from kiaomni_chat.engine import get_engine

    eng = get_engine()
    eng.ensure_loaded()

    # Load the Starship article
    article_path = REPO / "kiaomni_chat" / "static" / "sample_data" / "article.txt"
    article = article_path.read_text(encoding="utf-8")

    messages = [
        {"role": "user", "content": f"Read the article and answer Q1..Q9.\n\n{article}"},
    ]
    input_ids = eng.build_input_ids(messages)
    L = input_ids.shape[1]

    user_mask = eng._user_content_mask(input_ids, messages, L)
    user_ratio = float(user_mask.sum()) / float(L)
    assert user_ratio > 0.5, (
        f"User content should dominate the prompt (>50%), got {user_ratio:.1%}"
    )

    # Run the saliency-based selection
    keep, sal, n_kept = eng._select_keep_with_chat_template_mask(
        input_ids, messages, policy="kiaomni_gaussian", budget=512,
    )
    kept_user = int(np.isin(keep, np.where(user_mask)[0]).sum())
    kept_user_ratio = kept_user / len(keep)
    assert kept_user_ratio > 0.8, (
        f"After masking, >80% of kept tokens should be user content, "
        f"got {kept_user_ratio:.1%} ({kept_user}/{len(keep)})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_q1_scored_correctly_with_kiaomni_gaussian():
    """Q1 (NIAH single needle — IFT-3 launch date) must score 1.0
    with kiaomni_gaussian @ B=512 after the chat-template mask fix."""
    import numpy as np
    from kiaomni_chat.engine import get_engine
    from kiaomni_chat.grade_export import _score_tier1

    with open(REPO / "kiaomni_chat" / "sample_data" / "expected_answers.json") as f:
        expected = json.load(f)
    q1 = expected["questions"][0]  # tier 1, match: date

    eng = get_engine()
    eng.ensure_loaded()

    article_path = REPO / "kiaomni_chat" / "static" / "sample_data" / "article.txt"
    article = article_path.read_text(encoding="utf-8")

    messages = [
        {"role": "user", "content": f"Read the article and answer Q1..Q9.\n\n{article}"},
    ]
    gen = eng.generate_full(
        messages,
        policy="kiaomni_gaussian",
        budget=512,
        max_new_tokens=2048,
    )

    score, note, _ = _score_tier1(q1, gen.text)
    assert score == 1.0, f"Q1 should score 1.0, got {score}: {note}"
