#!/usr/bin/env python3
"""Validation: can claude-sonnet-5 reproduce Maya's human abstention labels?
Report-only. Reads abstention_validation_sample_Maya.csv.xlsx, sends each row's
raw_response + organ to claude-sonnet-5 (temp 0, independent calls), records a
binary label, and compares Sonnet-vs-human and detector-vs-human.

Needs ANTHROPIC_API_KEY in the environment. Writes a NEW CSV; modifies nothing.
"""
import argparse
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "abstention_validation_sample_Maya.csv.xlsx"
OUT = ROOT / "writing" / "abstention_validation_sonnet.csv"
MODEL = "claude-sonnet-5"

SYSTEM = (
    "You are labeling whether a model's response COMMITS to a verdict about a\n"
    "specific organ, or HEDGES (declines to commit).\n"
    "Label 0 (committed) if the response asserts the organ is present OR asserts it\n"
    "is absent, in plain language. Direction of the verdict does not matter, and\n"
    "whether the verdict is correct does not matter.\n"
    "Label 1 (hedged) if the response declines to determine the organ's status —\n"
    'e.g. "not clearly visible", "may represent", "cannot confirm", "possibly".\n'
    "Judge ONLY the organ named in the query. A hedge about a different organ in the\n"
    "same response does not make this organ hedged.\n"
    "Descriptive uses of cue words are NOT hedges: \"appears as a soft tissue\n"
    'structure" (describes radiodensity), "partially obscured by the liver"\n'
    '(describes the scene), "partially visible" (describes extent) are all 0.\n'
    '"appears to be present" alongside a direct assertion ("is seen") is 0.\n'
    "Respond with only the digit 0 or 1, nothing else."
)


def _parse_digit(txt):
    for ch in txt.strip():
        if ch in "01":
            return int(ch)
    return None


def label_row_api(client, organ, raw):
    """Anthropic API backend (needs ANTHROPIC_API_KEY + API credits)."""
    import anthropic
    user = f"Organ to judge: {organ}\n\nModel response:\n{raw}"
    for attempt in range(5):
        try:
            m = client.messages.create(
                model=MODEL, max_tokens=5, temperature=0,
                system=SYSTEM, messages=[{"role": "user", "content": user}])
            return _parse_digit("".join(b.text for b in m.content if b.type == "text"))
        except (anthropic.RateLimitError, anthropic.APIStatusError,
                anthropic.APIConnectionError) as e:
            wait = 2 ** attempt
            print(f"  [retry {attempt+1}] {type(e).__name__}; sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)
    return None


def label_row_cli(organ, raw):
    """Claude Code headless backend (uses the Claude subscription, no API key).
    Each `claude -p` invocation is a fresh, independent, stateless context.
    NOTE: the rubric is APPENDED to Claude Code's agent system prompt (it can't be
    fully replaced headlessly), so this is not as isolated as the API backend."""
    user = f"Organ to judge: {organ}\n\nModel response:\n{raw}"
    cmd = ["claude", "-p", "--model", "sonnet",
           "--append-system-prompt", SYSTEM,
           "--disallowedTools", "*"]
    for attempt in range(3):
        try:
            r = subprocess.run(cmd, input=user, capture_output=True, text=True, timeout=120)
            d = _parse_digit(r.stdout)
            if d is not None:
                return d
        except subprocess.TimeoutExpired:
            pass
        time.sleep(2 ** attempt)
    return None


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in (0, 1))
    return (po - pe) / (1 - pe) if pe != 1 else float("nan")


def confusion(rowlab, collab):
    # returns dict for a-vs-b as (a==1&b==1, a==1&b==0, a==0&b==1, a==0&b==0)
    c = Counter((int(x), int(y)) for x, y in zip(rowlab, collab))
    return c


def metrics_block(name, pred, human):
    acc = sum(p == h for p, h in zip(pred, human)) / len(human)
    k = kappa(pred, human)
    c = confusion(pred, human)
    TP, FP, FN, TN = c[(1, 1)], c[(1, 0)], c[(0, 1)], c[(0, 0)]
    prec = TP / (TP + FP) if TP + FP else float("nan")
    rec = TP / (TP + FN) if TP + FN else float("nan")
    print(f"  [{name} vs human]  acc={acc:.3f}  kappa={k:.3f}  "
          f"precision={prec:.3f}  recall={rec:.3f}")
    print(f"      confusion (pred rows x human cols): "
          f"TP(1,1)={TP} FP(1,0)={FP} FN(0,1)={FN} TN(0,0)={TN}")


def load_dotenv():
    """Minimal .env loader (repo root); no external dependency. Does not override
    a var already set in the environment."""
    envf = ROOT / ".env"
    if not envf.exists():
        return
    for line in envf.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["cli", "api"], default="cli",
                    help="cli = Claude Code subscription (no API key); api = Anthropic API")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only first N rows")
    a = ap.parse_args()

    client = None
    if a.backend == "api":
        load_dotenv()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY not set — create .env with ANTHROPIC_API_KEY=sk-ant-...")
        import anthropic
        client = anthropic.Anthropic()

    df = pd.read_excel(SRC)
    df["detector_flag"] = df["detector_flag"].astype(int)
    df["human_label"] = df["human_label"].astype(int)
    if a.limit:
        df = df.head(a.limit).copy()

    sonnet = []
    for i, r in enumerate(df.itertuples()):
        lab = (label_row_api(client, r.organ, r.raw_response) if a.backend == "api"
               else label_row_cli(r.organ, r.raw_response))
        sonnet.append(lab)
        print(f"  ...{i+1}/{len(df)} labeled (row={lab})", file=sys.stderr)
    df["sonnet_label"] = sonnet

    bad = df["sonnet_label"].isna().sum()
    if bad:
        print(f"WARNING: {bad} rows returned no parseable 0/1 (excluded from metrics)")
    good = df[df["sonnet_label"].notna()].copy()
    good["sonnet_label"] = good["sonnet_label"].astype(int)

    print(f"\n=== AGREEMENT (n={len(good)}) ===")
    metrics_block("Sonnet ", good["sonnet_label"].tolist(), good["human_label"].tolist())
    metrics_block("detector", good["detector_flag"].tolist(), good["human_label"].tolist())

    print("\n=== per-model ===")
    for m in ("medgemma27b", "qwen"):
        sub = good[good["model"] == m]
        if not len(sub):
            continue
        print(f"-- {m} (n={len(sub)}) --")
        metrics_block("Sonnet ", sub["sonnet_label"].tolist(), sub["human_label"].tolist())
        metrics_block("detector", sub["detector_flag"].tolist(), sub["human_label"].tolist())

    print("\n=== Sonnet vs Maya DISAGREEMENTS ===")
    dis = good[good["sonnet_label"] != good["human_label"]]
    print(f"{len(dis)} disagreements")
    for r in dis.itertuples():
        direction = ("Sonnet=1/Maya=0 (Sonnet says hedge, human says commit)"
                     if r.sonnet_label == 1 else
                     "Sonnet=0/Maya=1 (Sonnet says commit, human says hedge)")
        print(f"\n--- {r.sample_id} | {r.organ} | {r.model} | {direction} ---")
        print(f"    {str(r.raw_response)[:400]}")

    if a.limit:
        print(f"\n[smoke run, {a.limit} rows — not writing {OUT.name}]")
    else:
        df.to_csv(OUT, index=False)
        print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
