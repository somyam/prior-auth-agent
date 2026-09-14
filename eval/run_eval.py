"""Evaluation harness for policy-grounded prior authorization.

Combines deterministic assertions with an optional LLM-as-a-judge semantic
rubric and operational limits. The judge sees only the response and policy
excerpts that were provided to the evaluated agent.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import graph

CASES_PATH = Path(__file__).parent / "cases.jsonl"
RESULTS_PATH = Path(__file__).parent / "results.json"
MAX_LATENCY_SECONDS = 120.0
MAX_TOOL_CALLS = graph.MAX_AGENT_STEPS + 1  # Initial search plus agent follow-ups.


def load_cases():
    return [json.loads(line) for line in CASES_PATH.read_text().splitlines() if line.strip()]


def citation_groundedness(reasoning: str, policy_info: str, n: int = 4) -> float | None:
    if "POLICY CITATION:" not in reasoning or not policy_info:
        return None
    citation = reasoning.split("POLICY CITATION:", 1)[1].strip().strip('"').strip("[]")
    citation_words, policy_words = citation.lower().split(), policy_info.lower().split()
    if len(citation_words) < n:
        return 1.0 if " ".join(citation_words) in " ".join(policy_words) else 0.0
    citation_grams = {tuple(citation_words[i:i + n]) for i in range(len(citation_words) - n + 1)}
    policy_grams = {tuple(policy_words[i:i + n]) for i in range(len(policy_words) - n + 1)}
    return len(citation_grams & policy_grams) / len(citation_grams) if citation_grams else 0.0


def format_ok(reasoning: str) -> bool:
    return "DECISION:" in reasoning and "POLICY CITATION:" in reasoning


def semantic_judge(reasoning: str, policy_info: str, model_client) -> dict:
    """Score semantic support without requiring literal quote overlap."""
    prompt = f"""You are a strict evaluator of a policy-grounded authorization response.
Do not decide whether care is medically appropriate. Judge only whether the RESPONSE
is supported by the RETRIEVED POLICY EVIDENCE supplied to the responding agent.

RETRIEVED POLICY EVIDENCE:
{policy_info[:6000]}

RESPONSE:
{reasoning[:3000]}

Return only JSON in this exact shape:
{{"citation_entailment": 0.0, "decision_support": 0.0, "manual_review_appropriate": true, "verdict": "pass|fail", "rationale": "one sentence"}}

Scoring rubric:
- citation_entailment: whether the citation is semantically supported by the evidence.
- decision_support: whether the stated decision follows from the cited policy and documented facts in the response.
- manual_review_appropriate: true when a PENDED response reasonably follows from missing or ambiguous evidence.
- verdict: pass only when the response is grounded; a well-explained PENDED may pass."""
    try:
        raw = graph._call_model(model_client, prompt, temperature=0.0)
        result = graph._parse_json_object(raw)
        if not result:
            return {"error": "judge returned invalid JSON"}
        for metric in ("citation_entailment", "decision_support"):
            result[metric] = max(0.0, min(1.0, float(result.get(metric, 0.0))))
        result["verdict"] = result.get("verdict") if result.get("verdict") in {"pass", "fail"} else "fail"
        result["manual_review_appropriate"] = bool(result.get("manual_review_appropriate"))
        return result
    except Exception as exc:  # Keep deterministic results if the optional judge is unavailable.
        return {"error": str(exc)}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * p))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-semantic-judge", action="store_true", help="Skip LLM-as-a-judge calls.")
    args = parser.parse_args()
    agent = graph.build_graph()
    judge_client = None if args.skip_semantic_judge else graph.create_model_client()
    results = []

    for case in load_cases():
        start = time.perf_counter()
        try:
            state = agent.invoke({
                "patient_id": case["patient_id"],
                "diagnosis_code": case["diagnosis_code"],
                "procedure": case["procedure"],
            })
            elapsed = round(time.perf_counter() - start, 2)
            trace = state.get("tool_trace", [])
            row = {
                **case,
                "actual_decision": state["decision"],
                "response_time_seconds": elapsed,
                "tool_calls": len(trace),
                "agent_steps": state.get("agent_steps", 0),
                "operational_ok": elapsed <= MAX_LATENCY_SECONDS and len(trace) <= MAX_TOOL_CALLS,
                "format_ok": format_ok(state["reasoning"]),
                "citation_groundedness": citation_groundedness(state["reasoning"], state.get("policy_info", "")),
                "reasoning": state["reasoning"],
            }
            if not args.skip_semantic_judge and case["category"] != "auto_deny":
                row["semantic_judge"] = semantic_judge(state["reasoning"], state.get("policy_info", ""), judge_client)
            if case.get("expected_decision"):
                row["correct"] = row["actual_decision"] == case["expected_decision"]
        except Exception as exc:
            elapsed = round(time.perf_counter() - start, 2)
            row = {**case, "error": str(exc), "response_time_seconds": elapsed, "operational_ok": False}
        results.append(row)
        status = "ERROR" if row.get("error") else ("OK" if row.get("correct") else "--")
        print(f"[{case['category']:>10}] {case['id']:<12} actual={row.get('actual_decision', 'ERROR'):<8} {status}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    completed = [row for row in results if not row.get("error")]
    graded = [row for row in completed if "correct" in row]
    latencies = [row["response_time_seconds"] for row in completed]
    tool_calls = [row["tool_calls"] for row in completed]
    print("\n" + "=" * 60)
    if graded:
        accuracy = sum(row["correct"] for row in graded) / len(graded)
        print(f"graded accuracy: {sum(row['correct'] for row in graded)}/{len(graded)} ({accuracy:.0%})")
    if latencies:
        print(f"latency: mean={statistics.mean(latencies):.2f}s p50={percentile(latencies, .50):.2f}s p95={percentile(latencies, .95):.2f}s")
        print(f"tool calls: mean={statistics.mean(tool_calls):.1f} max={max(tool_calls)}")
        within_limits = sum(row["operational_ok"] for row in completed)
        print(f"operational limits: {within_limits}/{len(completed)} within {MAX_LATENCY_SECONDS:.0f}s and {MAX_TOOL_CALLS} tool calls")
        pended = sum(row.get("actual_decision") == "PENDED" for row in completed)
        print(f"manual-review rate: {pended}/{len(completed)} ({pended / len(completed):.0%})")
    judged = [row["semantic_judge"] for row in completed if row.get("semantic_judge") and not row["semantic_judge"].get("error")]
    if judged:
        print(f"semantic citation entailment: {statistics.mean(j['citation_entailment'] for j in judged):.2f}")
        print(f"semantic decision support: {statistics.mean(j['decision_support'] for j in judged):.2f}")
        print(f"semantic rubric pass rate: {sum(j['verdict'] == 'pass' for j in judged)}/{len(judged)}")
    if len(completed) != len(results):
        print(f"errors: {len(results) - len(completed)}")
    print(f"\nfull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
