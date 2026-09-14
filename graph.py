"""Bounded, policy-grounded prior-authorization investigation graph."""
import json
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from typing import Any, TypedDict

import boto3
import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

import audit
from index import build_index
from tools import lookup_diagnosis_fn, lookup_patient_fn, patient_section_fn, search_policy_fn

load_dotenv()
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
PATIENTS_CSV = os.path.join(os.path.dirname(__file__), "data", "patients.csv")
MODEL_ID = "us.meta.llama3-1-8b-instruct-v1:0"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
MAX_AGENT_STEPS = 4
ALLOWED_ACTIONS = {"search_policy", "inspect_patient", "finalize"}
ALLOWED_SECTIONS = {"conditions", "medications", "procedures"}
ALLOWED_DECISIONS = {"APPROVED", "DENIED", "PENDED"}


class PriorAuthState(TypedDict, total=False):
    patient_id: str
    diagnosis_code: str
    procedure: str
    start_time: str
    diagnosis_info: str
    patient_info: str
    patient_found: bool
    policy_info: str
    tool_trace: list[dict[str, Any]]
    agent_steps: int
    agent_action: dict[str, Any]
    assessment: dict[str, Any]
    decision: str
    reasoning: str
    response_time_seconds: float


def create_model_client():
    """Create the optional remote client; local Ollama needs no Python client."""
    if LLM_PROVIDER == "bedrock":
        return boto3.client(
            service_name="bedrock-runtime", region_name=AWS_REGION,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
        )
    if LLM_PROVIDER == "ollama":
        return None
    raise ValueError("LLM_PROVIDER must be either 'ollama' or 'bedrock'.")


def _load_components():
    model_client = create_model_client()
    embedding_model, faiss_index, chunks, sources = build_index()
    return model_client, embedding_model, faiss_index, chunks, sources, pd.read_csv(PATIENTS_CSV)


def _call_model(model_client, prompt: str, temperature: float = 0.1) -> str:
    """Generate with the configured local Ollama server or optional Bedrock backend."""
    if LLM_PROVIDER == "bedrock":
        body = json.dumps({
            "prompt": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
            "max_gen_len": 750, "temperature": temperature,
        })
        response = model_client.invoke_model(modelId=MODEL_ID, body=body)
        return json.loads(response["body"].read())["generation"]

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 750},
    }).encode("utf-8")
    request = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read())["response"]
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL}. Start Ollama and pull '{OLLAMA_MODEL}', "
            "or set LLM_PROVIDER=bedrock to use AWS Bedrock."
        ) from exc


def _is_degenerate(text: str, n: int = 6, max_repeats: int = 4) -> bool:
    words = text.split()
    if len(words) < n * max_repeats:
        return False
    counts = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return counts.most_common(1)[0][1] > max_repeats


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Accept a JSON object even if a small model wraps it in a code fence."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _agent_response(model_client, prompt: str) -> dict[str, Any] | None:
    for attempt in range(3):
        raw = _call_model(model_client, prompt, temperature=0.1 + (attempt * 0.2))
        parsed = _parse_json_object(raw)
        if parsed and not _is_degenerate(raw):
            return parsed
    return None


def _manual_review(reason: str) -> dict[str, Any]:
    return {"action": "finalize", "decision": "PENDED", "reasoning": reason, "criteria": []}


def _format_assessment(assessment: dict[str, Any]) -> str:
    citations = [item.get("policy_citation", "") for item in assessment.get("criteria", [])]
    citation = " | ".join(dict.fromkeys(item for item in citations if item)) or "none — manual review"
    return f"DECISION: {assessment['decision']}\nREASONING: {assessment.get('reasoning', 'No rationale provided.')}\nPOLICY CITATION: {citation}"


def build_graph():
    """Create a bounded agent loop over read-only patient and policy evidence."""
    model_client, embedding_model, faiss_index, chunks, sources, patients = _load_components()

    def node_start(_: PriorAuthState) -> dict:
        return {"start_time": datetime.now().isoformat(), "tool_trace": [], "agent_steps": 0}

    def node_lookup_diagnosis(state: PriorAuthState) -> dict:
        return {"diagnosis_info": lookup_diagnosis_fn(state["diagnosis_code"])}

    def node_lookup_patient(state: PriorAuthState) -> dict:
        patient_info = lookup_patient_fn(patients, state["patient_id"])
        return {"patient_info": patient_info or "", "patient_found": patient_info is not None}

    def route_after_patient(state: PriorAuthState) -> str:
        return "initial_policy_search" if state["patient_found"] else "unknown_patient"

    def node_unknown_patient(_: PriorAuthState) -> dict:
        assessment = _manual_review("No matching patient record was found; a verified history is required.")
        assessment["decision"] = "DENIED"
        return {"policy_info": "", "assessment": assessment, "decision": "DENIED", "reasoning": _format_assessment(assessment)}

    def node_initial_policy_search(state: PriorAuthState) -> dict:
        query = f"{state['procedure']} {state['diagnosis_info']} coverage criteria medical necessity"
        result = search_policy_fn(query, embedding_model, faiss_index, chunks, sources)
        return {"policy_info": result, "tool_trace": [{"tool": "search_policy", "query": query, "result": result}]}

    def node_agent(state: PriorAuthState) -> dict:
        forced_finalize = state["agent_steps"] >= MAX_AGENT_STEPS
        prompt = f"""You are a Medicare prior-authorization evidence investigator.
Use only the case facts and policy excerpts below. You may not use outside knowledge,
invent documentation, approve with missing criteria, or call tools not listed.

CASE
Patient: {state['patient_info']}
Diagnosis: {state['diagnosis_info']}
Requested procedure: {state['procedure']}

POLICY EVIDENCE
{state.get('policy_info', '')[:6000]}

TOOL TRACE
{json.dumps(state.get('tool_trace', [])[-4:])[:5000]}

Choose exactly one next action and return ONLY a JSON object.
Allowed actions:
1. {{"action":"search_policy","query":"specific policy requirement to retrieve"}}
2. {{"action":"inspect_patient","section":"conditions|medications|procedures"}}
3. {{"action":"finalize","decision":"APPROVED|DENIED|PENDED","reasoning":"plain-language rationale","criteria":[{{"criterion":"coverage requirement","status":"met|not_met|not_documented","evidence":"case fact or missing fact","policy_citation":"exact quote from POLICY EVIDENCE"}}]}}

Decision rules: APPROVED requires one or more criteria and every criterion must be met.
DENIED requires at least one criterion marked not_met. PENDED is required when a needed
criterion is not_documented, evidence is ambiguous, or policy evidence is insufficient.
{('You have reached the tool limit: action must be finalize.' if forced_finalize else '')}"""
        action = _agent_response(model_client, prompt) or _manual_review("The evidence agent did not return a valid structured response.")
        if forced_finalize and action.get("action") != "finalize":
            action = _manual_review("The evidence-agent tool limit was reached before a supported decision.")
        if action.get("action") not in ALLOWED_ACTIONS:
            action = _manual_review("The evidence agent attempted an unsupported action.")
        return {"agent_action": action}

    def route_after_agent(state: PriorAuthState) -> str:
        return "execute_tool" if state["agent_action"].get("action") != "finalize" else "verify"

    def node_execute_tool(state: PriorAuthState) -> dict:
        action, trace = state["agent_action"], list(state.get("tool_trace", []))
        if action["action"] == "search_policy":
            query = str(action.get("query", "")).strip()[:300]
            result = search_policy_fn(query, embedding_model, faiss_index, chunks, sources) if query else "Tool error: a policy-search query is required."
            trace.append({"tool": "search_policy", "query": query, "result": result})
            return {"policy_info": f"{state.get('policy_info', '')}\n\n--- FOLLOW-UP SEARCH ---\n{result}", "tool_trace": trace, "agent_steps": state["agent_steps"] + 1}
        section = str(action.get("section", "")).lower()
        result = patient_section_fn(patients, state["patient_id"], section) if section in ALLOWED_SECTIONS else None
        trace.append({"tool": "inspect_patient", "section": section, "result": result or "Tool error: unsupported or empty patient section."})
        return {"tool_trace": trace, "agent_steps": state["agent_steps"] + 1}

    def node_verify(state: PriorAuthState) -> dict:
        assessment = state["agent_action"]
        decision, criteria = str(assessment.get("decision", "")).upper(), assessment.get("criteria")
        if decision not in ALLOWED_DECISIONS or not isinstance(criteria, list):
            assessment = _manual_review("The evidence agent returned an invalid final assessment.")
        else:
            statuses, valid, policy = [], bool(criteria), state.get("policy_info", "").lower()
            for item in criteria:
                if not isinstance(item, dict):
                    valid = False
                    break
                status, quote = item.get("status"), str(item.get("policy_citation", "")).strip()
                statuses.append(status)
                if status not in {"met", "not_met", "not_documented"} or not quote or quote.lower() not in policy:
                    valid = False
            if decision == "APPROVED" and (not valid or any(status != "met" for status in statuses)):
                assessment = _manual_review("Approval was not fully supported by documented evidence and retrieved policy text.")
            elif decision == "DENIED" and (not valid or "not_met" not in statuses):
                assessment = _manual_review("Denial was not tied to a documented unmet policy criterion.")
            elif decision == "PENDED" and not valid:
                assessment = _manual_review("Policy evidence was insufficient for a verifiable assessment.")
        return {"assessment": assessment, "decision": assessment["decision"], "reasoning": _format_assessment(assessment)}

    def node_audit_log(state: PriorAuthState) -> dict:
        elapsed = (datetime.now() - datetime.fromisoformat(state["start_time"])).total_seconds()
        audit.log_decision({
            "patient_id": state["patient_id"], "diagnosis_code": state["diagnosis_code"], "procedure": state["procedure"],
            "decision": state["decision"], "reasoning": state["reasoning"], "timestamp": state["start_time"],
            "response_time_seconds": elapsed, "tool_trace": state.get("tool_trace", []),
            "assessment": state.get("assessment", {}), "policy_info": state.get("policy_info", ""),
        })
        return {"response_time_seconds": elapsed}

    builder = StateGraph(PriorAuthState)
    for name, node in {
        "start": node_start, "lookup_diagnosis": node_lookup_diagnosis, "lookup_patient": node_lookup_patient,
        "unknown_patient": node_unknown_patient, "initial_policy_search": node_initial_policy_search,
        "agent": node_agent, "execute_tool": node_execute_tool, "verify": node_verify, "audit": node_audit_log,
    }.items():
        builder.add_node(name, node)
    builder.add_edge(START, "start")
    builder.add_edge("start", "lookup_diagnosis")
    builder.add_edge("lookup_diagnosis", "lookup_patient")
    builder.add_conditional_edges("lookup_patient", route_after_patient, {"initial_policy_search": "initial_policy_search", "unknown_patient": "unknown_patient"})
    builder.add_edge("initial_policy_search", "agent")
    builder.add_conditional_edges("agent", route_after_agent, {"execute_tool": "execute_tool", "verify": "verify"})
    builder.add_edge("execute_tool", "agent")
    builder.add_edge("verify", "audit")
    builder.add_edge("unknown_patient", "audit")
    builder.add_edge("audit", END)
    return builder.compile()
