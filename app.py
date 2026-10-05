"""Streamlit UI for the prior authorization agent."""
import uuid

import pandas as pd
import streamlit as st
from langgraph.types import Command

import audit
from graph import PATIENTS_CSV, build_graph

st.set_page_config(page_title="Prior Auth AI Agent", page_icon="⚕", layout="centered")

DIAGNOSIS_OPTIONS = {
    "M54.5 — Low back pain": "M54.5",
    "M51.1 — Lumbar disc herniation with radiculopathy": "M51.1",
    "M48.06 — Spinal stenosis, lumbar region": "M48.06",
    "M54.4 — Lumbago with sciatica": "M54.4",
    "M17.1 — Primary osteoarthritis of knee": "M17.1",
    "M17.0 — Bilateral primary osteoarthritis of knee": "M17.0",
    "Z12.11 — Colon cancer screening": "Z12.11",
    "K92.1 — Melena (blood in stool)": "K92.1",
}
PROCEDURES = [
    "MRI of the lumbar spine",
    "Total knee replacement surgery",
    "Screening colonoscopy",
    "MRI of the knee",
    "Physical therapy for back pain",
]


@st.cache_resource
def load_graph():
    return build_graph()


@st.cache_data
def load_patients():
    return pd.read_csv(PATIENTS_CSV)


st.title("⚕ Prior Authorization AI Agent")
st.caption("All patient data is synthetic.")

graph = load_graph()
patients = load_patients()

with st.form("prior_auth_form"):
    patient_id = st.selectbox("Patient", patients["patient_id"].tolist())
    diagnosis_label = st.selectbox("Diagnosis", list(DIAGNOSIS_OPTIONS.keys()))
    procedure = st.selectbox("Requested procedure", PROCEDURES)
    submitted = st.form_submit_button("Run prior authorization review")

if submitted:
    diagnosis_code = DIAGNOSIS_OPTIONS[diagnosis_label]
    thread_id = str(uuid.uuid4())
    with st.spinner("Reviewing against CMS policy..."):
        result = graph.invoke(
            {"patient_id": patient_id, "diagnosis_code": diagnosis_code, "procedure": procedure},
            {"configurable": {"thread_id": thread_id}},
        )

    if result.get("__interrupt__"):
        audit.queue_for_review(thread_id, patient_id, diagnosis_code, procedure, result["reasoning"])
        st.warning(f"PENDED — queued for manual review (case {thread_id[:8]}).")
        st.write(result["reasoning"])
    else:
        if result["decision"] == "APPROVED":
            st.success(result["decision"])
        else:
            st.error(result["decision"])
        st.write(result["reasoning"])
        st.caption(f"⏱ {result['response_time_seconds']:.2f}s")

st.divider()
st.subheader("Pending manual reviews")
pending = audit.fetch_pending_reviews()
if pending:
    options = {f"{pid} · {dx} · {proc} (queued {queued_at})": tid for tid, pid, dx, proc, _, queued_at in pending}
    choice = st.selectbox("Case awaiting review", list(options.keys()))
    thread_id, _, _, _, reasoning, _ = next(row for row in pending if row[0] == options[choice])
    st.write(reasoning)
    with st.form("review_form"):
        review_decision = st.radio("Reviewer decision", ["APPROVED", "DENIED", "PENDED"], horizontal=True)
        reviewer_notes = st.text_input("Reviewer notes (optional)")
        review_submitted = st.form_submit_button("Submit review")
    if review_submitted:
        final_state = graph.invoke(
            Command(resume={"decision": review_decision, "reviewer_notes": reviewer_notes}),
            {"configurable": {"thread_id": thread_id}},
        )
        audit.resolve_review(thread_id)
        st.success(f"Recorded: {final_state['decision']}")
        st.rerun()
else:
    st.caption("No cases awaiting review.")

st.divider()
st.subheader("Audit log")
rows = audit.fetch_all()
if rows:
    st.dataframe(pd.DataFrame(rows, columns=[
        "patient_id", "diagnosis_code", "procedure", "decision", "timestamp", "response_time_seconds",
    ]))
else:
    st.caption("No decisions logged yet.")

st.caption("All patient data is synthetic. No real PHI used at any stage.")
