#!/usr/bin/env python3
"""Entity-level evaluation of PHI-Shield against the synthetic gold set.

Matching rule (documented, partial-credit-free): a predicted entity is a
true positive for a gold entity when

  * the entity types are identical, and
  * the predicted span covers at least 50% of the gold span
    (intersection length >= 0.5 * gold length).

Matching is greedy one-to-one (predictions in document order).  Metrics
are entity-level precision / recall / F1, per PHI type and micro-averaged
overall.  Headline metric: overall recall.

Outputs eval_report.md next to this file and prints the same summary.

Run:  python evaluate.py [--data data/synthetic_notes.jsonl]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from phi_shield.engine import analyze, build_analyzer


def spans_match(gold, pred):
    if gold["entity_type"] != pred["entity_type"]:
        return False
    inter = max(0, min(gold["end"], pred["end"]) - max(gold["start"], pred["start"]))
    return inter >= 0.5 * (gold["end"] - gold["start"])


def score_records(records, analyzer):
    """Return per-type and overall counts: {type: {tp, fp, fn}}."""
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for record in records:
        gold = list(record["entities"])
        preds = [
            {"start": r.start, "end": r.end, "entity_type": r.entity_type}
            for r in analyze(analyzer, record["text"])
        ]
        used_pred = set()
        for g in gold:
            hit = None
            for i, p in enumerate(preds):
                if i in used_pred:
                    continue
                if spans_match(g, p):
                    hit = i
                    break
            if hit is None:
                counts[g["entity_type"]]["fn"] += 1
            else:
                used_pred.add(hit)
                counts[g["entity_type"]]["tp"] += 1
        for i, p in enumerate(preds):
            if i not in used_pred:
                counts[p["entity_type"]]["fp"] += 1
    return counts


def prf(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def render_report(counts, backend, n_notes):
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for c in counts.values():
        for k in totals:
            totals[k] += c[k]
    op, orr, of1 = prf(totals)
    lines = [
        "# PHI-Shield evaluation report",
        "",
        f"Dataset: {n_notes} synthetic clinical notes (data/synthetic_notes.jsonl). "
        "All data synthetic; no real patient data.",
        f"Analyzer backend: {backend}",
        "",
        "Matching rule: same entity type and predicted span covers >= 50% "
        "of the gold span; greedy one-to-one matching.",
        "",
        f"**Overall precision: {op:.3f} | Overall recall: {orr:.3f} | "
        f"Overall F1: {of1:.3f}**",
        "",
        "| Entity type | Gold | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for etype in sorted(counts):
        c = counts[etype]
        p, r, f1 = prf(c)
        gold_n = c["tp"] + c["fn"]
        lines.append(
            f"| {etype} | {gold_n} | {c['tp']} | {c['fp']} | {c['fn']} "
            f"| {p:.3f} | {r:.3f} | {f1:.3f} |"
        )
    lines.append(
        f"| **OVERALL** | {totals['tp'] + totals['fn']} | {totals['tp']} "
        f"| {totals['fp']} | {totals['fn']} | {op:.3f} | {orr:.3f} | {of1:.3f} |"
    )
    lines.append("")
    return "\n".join(lines) + "\n", (op, orr, of1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(Path(__file__).resolve().parent / "data" / "synthetic_notes.jsonl"),
    )
    parser.add_argument(
        "--report",
        default=str(Path(__file__).resolve().parent / "eval_report.md"),
    )
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    analyzer, backend = build_analyzer()
    counts = score_records(records, analyzer)
    report, (p, r, f1) = render_report(counts, backend, len(records))
    Path(args.report).write_text(report, encoding="utf-8")
    print(report)
    print(f"(report written to {args.report})")


if __name__ == "__main__":
    main()
