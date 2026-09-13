"""Lookup and retrieval functions used by the prior-auth graph nodes."""
import numpy as np
import pandas as pd

ICD10_MAP = {
    "M54.5": "Low back pain",
    "M54.4": "Lumbago with sciatica",
    "M51.1": "Lumbar disc herniation with radiculopathy",
    "M17.1": "Primary osteoarthritis of knee",
    "M17.0": "Bilateral primary osteoarthritis of knee",
    "Z12.11": "Encounter for screening for colon cancer",
    "K92.1": "Melena — blood in stool",
    "K57.30": "Diverticulosis of large intestine",
    "M47.816": "Spondylosis with radiculopathy, lumbar",
    "M48.06": "Spinal stenosis, lumbar region",
    "Z80.0": "Family history of malignant neoplasm of digestive organs",
    "K57.32": "Diverticulitis of large intestine without abscess",
}


def lookup_diagnosis_fn(code: str) -> str:
    """Look up what an ICD-10 diagnosis code means."""
    code = code.strip().upper()
    if code in ICD10_MAP:
        return f"ICD-10 {code} = {ICD10_MAP[code]}"
    return f"ICD-10 {code} = Valid diagnosis code."


def lookup_patient_fn(patients: pd.DataFrame, patient_id: str) -> str | None:
    """Look up a patient's medical history. Returns None if the ID isn't found."""
    match = patients[patients["patient_id"] == patient_id]
    if match.empty:
        return None
    row = match.iloc[0]
    return (
        f"Patient ID: {row['patient_id']}\n"
        f"Gender: {row['gender']}\n"
        f"Conditions: {row['conditions']}\n"
        f"Medications: {row['medications']}\n"
        f"Procedures: {row['procedures']}"
    )


def patient_section_fn(patients: pd.DataFrame, patient_id: str, section: str) -> str | None:
    """Return one whitelisted part of a record for the bounded evidence agent."""
    match = patients[patients["patient_id"] == patient_id]
    if match.empty or section not in {"conditions", "medications", "procedures"}:
        return None
    value = match.iloc[0][section]
    return f"Patient {section}: {value if pd.notna(value) else 'not documented'}"


def search_policy_fn(query: str, embedding_model, faiss_index, chunks, sources, k: int = 3) -> str:
    """Search the CMS policy FAISS index for the top-k most relevant chunks."""
    query_vector = embedding_model.encode([query])
    _, indices = faiss_index.search(np.array(query_vector, dtype=np.float32), k)
    results = [
        f"Source: {sources[idx]}\nContent: {chunks[idx][:500]}"
        for idx in indices[0]
    ]
    return "\n\n---\n\n".join(results)
