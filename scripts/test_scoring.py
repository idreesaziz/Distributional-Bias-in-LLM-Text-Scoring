"""
Dry-run test: send 3 samples to Google standard API to verify scoring works.
"""

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from src.llm_scoring import parse_score, SYSTEM_PROMPT, _call_with_retry


def load_test_samples(n=3):
    """Load n degraded samples — pick varied degradation levels."""
    deg_path = Path("data/degraded/degraded_samples.json")
    with open(deg_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    by_level = {}
    for s in samples:
        lvl = s.get("level", 0)
        by_level.setdefault(lvl, []).append(s)

    levels = sorted(by_level.keys())
    picks = []
    for lvl in [levels[0], levels[len(levels) // 2], levels[-1]]:
        picks.append(by_level[lvl][0])
        if len(picks) >= n:
            break

    return picks


def test_google(samples, api_key=None):
    """Test Google standard API with a few calls."""
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("[SKIP] GOOGLE_API_KEY not set")
        return

    model_id = "gemini-3-flash-preview"
    print(f"[Google] Sending {len(samples)} requests to {model_id}...")

    for i, sample in enumerate(samples):
        user_prompt = (
            f"Please rate the quality of the following text:\n\n"
            f"---\n{sample['degraded_text'][:500]}\n---"
        )
        try:
            raw = _call_with_retry("google", model_id, user_prompt, api_key=key)
            score = parse_score(raw)
            lvl = sample.get("level", "?")
            axis = sample.get("axis", "?")
            print(f"  Sample {i+1} (axis={axis}, level={lvl}): "
                  f"score={score}, raw='{raw.strip()}'")
        except Exception as e:
            print(f"  Sample {i+1}: ERROR {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", help="Google API key (or set GOOGLE_API_KEY)")
    args = parser.parse_args()

    samples = load_test_samples(3)
    print(f"Loaded {len(samples)} test samples\n")
    for s in samples:
        print(f"  id={s['id']}, axis={s.get('axis','?')}, "
              f"level={s.get('level','?')}, "
              f"text_len={len(s['degraded_text'])}")
    print()

    test_google(samples, api_key=args.key)
    print("\nDone.")
