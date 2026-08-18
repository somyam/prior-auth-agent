"""LangGraph pipeline: diagnosis lookup -> patient lookup -> policy search -> LLM decision -> audit log."""
import json
import os
from collections import Counter
from datetime import datetime
from typing import TypedDict

import boto3
import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

import audit
from index import build_index
from tools import lookup_diagnosis_fn, lookup_patient_fn, search_policy_fn

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
PATIENTS_CSV = os.path.join(os.path.dirname(__file__), "data", "patients.csv")
MODEL_ID = "us.meta.llama3-1-8b-instruct-v1:0"


class PriorAuthState(TypedDict):
    patient_id: str
    diagnosis_code: str
    procedure: str
    start_time: str
    diagnosis_info: str
    patient_info: str
    patient_found: bool
    policy_info: str
    decision: str
    reasoning: str
    response_time_seconds: float


def _load_components():
    bedrock = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    )
    embedding_model, faiss_index, chunks, sources = build_index()
    patients = pd.read_csv(PATIENTS_CSV)
    return bedrock, embedding_model, faiss_index, chunks, sources, patients


def _call_llama(bedrock, prompt: str, temperature: float = 0.1) -> str:
    body = json.dumps({
        "prompt": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        "max_gen_len": 600,
        "temperature": temperature,
    })
    response = bedrock.invoke_model(modelId=MODEL_ID, body=body)
    return json.loads(response["body"].read())["generation"]


def _is_degenerate(text: str, n: int = 6, max_repeats: int = 4) -> bool:
    """Llama 3.1 8B occasionally gets stuck cycling the same phrase at low temperature,
    burning its whole generation budget without reaching a POLICY CITATION. Flag it by
    checking whether any 6-word run repeats more than a few times."""
    words = text.split()
    if len(words) < n * max_repeats:
        return False
    counts = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return counts.most_common(1)[0][1] > max_repeats


def _decide_with_retry(bedrock, prompt: str, attempts: int = 3) -> str:
    """Retry with escalating temperature to break out of a repetition loop before
    giving up on a well-formed, policy-cited response."""
    response = ""
    for attempt in range(attempts):
        response = _call_llama(bedrock, prompt, temperature=0.1 + 0.3 * attempt)
        if "POLICY CITATION:" in response and not _is_degenerate(response):
            return response
    return response


def build_graph():
    """Load components once, close over them in the node closures, and compile the graph."""
    bedrock, embedding_model, faiss_index, chunks, sources, patients = _load_components()

    def node_start(state: PriorAuthState) -> dict:
        return {"start_time": datetime.now().isoformat()}

    def node_lookup_diagnosis(state: PriorAuthState) -> dict:
        return {"diagnosis_info": lookup_diagnosis_fn(state["diagnosis_code"])}

    def node_lookup_patient(state: PriorAuthState) -> dict:
        patient_info = lookup_patient_fn(patients, state["patient_id"])
        return {"patient_info": patient_info or "", "patient_found": patient_info is not None}

    def route_after_patient(state: PriorAuthState) -> str:
        return "search_policy" if state["patient_found"] else "unknown_patient"

    def node_unknown_patient(state: PriorAuthState) -> dict:
        return {
            "policy_info": "",
            "decision": "DENIED",
            "reasoning": (
                "No matching patient record was found; the request cannot be "
                "reviewed without a verified patient history."
            ),
        }

    def node_search_policy(state: PriorAuthState) -> dict:
        query = f"{state['procedure']} coverage criteria medical necessity"
        policy_info = search_policy_fn(query, embedding_model, faiss_index, chunks, sources)
        return {"policy_info": policy_info}

    def node_decide(state: PriorAuthState) -> dict:
        prompt = f"""You are reviewing a Medicare prior authorization request. Your only
job is to decide APPROVED or DENIED, using nothing but the coverage policy
excerpt below — do not rely on outside knowledge of Medicare rules.

CASE FILE
Patient: {state['patient_info']}
Diagnosis: {state['diagnosis_info']}
Procedure requested: {state['procedure']}

COVERAGE POLICY EXCERPT (the only source you may cite)
{state['policy_info'][:2000]}

Work through this in order:
1. Identify the coverage condition(s) in the excerpt that apply to this procedure.
2. Check the case file against each condition — met, not met, or not documented.
3. Decide APPROVED only if the documented case file satisfies the conditions; otherwise DENIED.
4. Quote the exact clause from the excerpt that drove your decision.

Respond in exactly this shape, nothing else:
DECISION: APPROVED or DENIED
REASONING: two to three sentences, plain clinical language
POLICY CITATION: the exact clause quoted above that the decision rests on
"""
        response = _decide_with_retry(bedrock, prompt)
        if "POLICY CITATION:" not in response or _is_degenerate(response):
            return {
                "decision": "DENIED",
                "reasoning": (
                    "DECISION: DENIED\n"
                    "REASONING: The model could not produce a well-formed, policy-cited "
                    "decision after multiple attempts. Denied by default and flagged for "
                    "manual review.\n"
                    "POLICY CITATION: none — automatic fallback"
                ),
            }
        decision = "APPROVED" if "APPROVED" in response.upper() else "DENIED"
        return {"decision": decision, "reasoning": response}

    def node_audit_log(state: PriorAuthState) -> dict:
        start_time = datetime.fromisoformat(state["start_time"])
        response_time_seconds = (datetime.now() - start_time).total_seconds()
        audit.log_decision({
            "patient_id": state["patient_id"],
            "diagnosis_code": state["diagnosis_code"],
            "procedure": state["procedure"],
            "decision": state["decision"],
            "reasoning": state["reasoning"],
            "timestamp": state["start_time"],
            "response_time_seconds": response_time_seconds,
        })
        return {"response_time_seconds": response_time_seconds}

    builder = StateGraph(PriorAuthState)
    builder.add_node("start", node_start)
    builder.add_node("lookup_diagnosis", node_lookup_diagnosis)
    builder.add_node("lookup_patient", node_lookup_patient)
    builder.add_node("unknown_patient", node_unknown_patient)
    builder.add_node("search_policy", node_search_policy)
    builder.add_node("decide", node_decide)
    builder.add_node("audit", node_audit_log)

    builder.add_edge(START, "start")
    builder.add_edge("start", "lookup_diagnosis")
    builder.add_edge("lookup_diagnosis", "lookup_patient")
    builder.add_conditional_edges("lookup_patient", route_after_patient, {
        "search_policy": "search_policy",
        "unknown_patient": "unknown_patient",
    })
    builder.add_edge("search_policy", "decide")
    builder.add_edge("decide", "audit")
    builder.add_edge("unknown_patient", "audit")
    builder.add_edge("audit", END)

    return builder.compile()
