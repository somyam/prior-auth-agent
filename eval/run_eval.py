"""Evaluation harness for the prior-auth agent.

This synthetic patient dataset has no field for the clinical details (symptom
duration, "red flag" signs, conservative-therapy trials, screening age/interval)
that the real CMS LCD criteria hinge on, so there's no way to know the "true"
clinically-correct decision for a general request. Instead this measures things
that ARE independently verifiable:

  1. auto_deny   - an unknown patient ID must be denied (deterministic routing,
                    not LLM-dependent)
  2. mismatch    - a diagnosis that has nothing to do with the requested
                    procedure (e.g. a GI diagnosis requesting a knee
                    replacement) must be denied, regardless of patient history
  3. aligned     - diagnosis and procedure are in the same clinical category;
                    the "correct" decision isn't knowable from this data, so
                    these are ungraded for accuracy, but still checked for
                    citation groundedness and output format

Citation groundedness: what fraction of the model's cited "POLICY CITATION"
text is verbatim 4-gram-overlapping with the policy chunks that were actually
retrieved and put in its prompt. It's a hallucination proxy, not a semantic
correctness check -- a paraphrased-but-accurate citation would score low too.
"""
import json
import time
from pathlib import Path

from graph import build_graph

CASES_PATH = Path(__file__).parent / "cases.jsonl"
RESULTS_PATH = Path(__file__).parent / "results.json"


def load_cases():
    return [json.loads(line) for line in CASES_PATH.read_text().splitlines() if line.strip()]


def citation_groundedness(reasoning: str, policy_info: str, n: int = 4) -> float | None:
    if "POLICY CITATION:" not in reasoning or not policy_info:
        return None
    citation = reasoning.split("POLICY CITATION:", 1)[1].strip().strip('"').strip("[]")
    words_c = citation.lower().split()
    words_p = policy_info.lower().split()
    if len(words_c) < n:
        return 1.0 if " ".join(words_c) in " ".join(words_p) else 0.0
    grams_c = {tuple(words_c[i:i + n]) for i in range(len(words_c) - n + 1)}
    grams_p = {tuple(words_p[i:i + n]) for i in range(len(words_p) - n + 1)}
    return len(grams_c & grams_p) / len(grams_c) if grams_c else 0.0


def format_ok(reasoning: str) -> bool:
    return "DECISION:" in reasoning and "POLICY CITATION:" in reasoning


def main():
    cases = load_cases()
    graph = build_graph()

    results = []
    for case in cases:
        start = time.time()
        state = graph.invoke({
            "patient_id": case["patient_id"],
            "diagnosis_code": case["diagnosis_code"],
            "procedure": case["procedure"],
        })
        elapsed = round(time.time() - start, 2)

        row = {**case, "actual_decision": state["decision"], "response_time_seconds": elapsed}
        if case["category"] != "auto_deny":
            row["format_ok"] = format_ok(state["reasoning"])
            row["citation_groundedness"] = citation_groundedness(state["reasoning"], state.get("policy_info", ""))
        if case.get("expected_decision"):
            row["correct"] = row["actual_decision"] == case["expected_decision"]
        row["reasoning"] = state["reasoning"]
        results.append(row)

        status = "OK" if row.get("correct") else ("--" if "correct" not in row else "FAIL")
        print(f"[{case['category']:>10}] {case['id']:<12} expected={str(case.get('expected_decision')):<8} actual={row['actual_decision']:<8} {status}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 60)
    graded = [r for r in results if "correct" in r]
    for cat in ["auto_deny", "mismatch"]:
        cat_rows = [r for r in graded if r["category"] == cat]
        if cat_rows:
            n_correct = sum(r["correct"] for r in cat_rows)
            print(f"{cat:>10} accuracy: {n_correct}/{len(cat_rows)} ({n_correct / len(cat_rows):.0%})")
    if graded:
        n_correct = sum(r["correct"] for r in graded)
        print(f"{'overall':>10} accuracy: {n_correct}/{len(graded)} ({n_correct / len(graded):.0%})")

    grounded = [r["citation_groundedness"] for r in results if r.get("citation_groundedness") is not None]
    formatted = [r["format_ok"] for r in results if "format_ok" in r]
    if grounded:
        print(f"\navg citation groundedness: {sum(grounded) / len(grounded):.2f} (n={len(grounded)})")
    if formatted:
        print(f"output format compliance: {sum(formatted) / len(formatted):.0%} (n={len(formatted)})")

    print(f"\nfull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
