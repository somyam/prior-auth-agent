"""Streamlit UI for the prior authorization agent."""
import pandas as pd
import streamlit as st

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
st.caption("RAG-grounded decisions over real CMS Medicare policy. All patient data is synthetic.")

graph = load_graph()
patients = load_patients()

with st.form("prior_auth_form"):
    patient_id = st.selectbox("Patient", patients["patient_id"].tolist())
    diagnosis_label = st.selectbox("Diagnosis", list(DIAGNOSIS_OPTIONS.keys()))
    procedure = st.selectbox("Requested procedure", PROCEDURES)
    submitted = st.form_submit_button("Run prior authorization review")

if submitted:
    with st.spinner("Reviewing against CMS policy..."):
        result = graph.invoke({
            "patient_id": patient_id,
            "diagnosis_code": DIAGNOSIS_OPTIONS[diagnosis_label],
            "procedure": procedure,
        })

    if result["decision"] == "APPROVED":
        st.success(result["decision"])
    else:
        st.error(result["decision"])
    st.write(result["reasoning"])
    st.caption(f"⏱ {result['response_time_seconds']:.2f}s")

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
