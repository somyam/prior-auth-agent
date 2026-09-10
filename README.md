# Prior Auth Agent

A RAG prior authorization agent that decides whether a requested procedure meets Medicare coverage criteria and produces an auditable, citation-backed rationale for every decision.

Grounded retrieval. Three CMS Local Coverage Determinations are parsed, chunked, embedded with a sentence-transformer from Hugging Face, and indexed in FAISS (disk-cached after first build). At decision time the top-matching policy passages for the requested procedure are injected as the only permissible source, and the model must return a fixed DECISION / REASONING / POLICY CITATION structure quoting the exact clause it relied on. Citations are scored for verbatim groundedness against the retrieved context as a hallucination check.

Deterministic control flow. The workflow is a LangGraph state machine — diagnosis lookup, patient lookup, retrieval, decision, audit. An unknown patient ID is denied and logged before any inference runs, so the model never reasons over an unverified record. Every outcome, auto-denials included, is persisted to a SQLite audit log.

Failure handling. The eval harness surfaced a repetition-loop failure mode in Llama 3.1 8B where the model exhausts its token budget without producing a citation. The system detects degenerate output, retries with escalating temperature, and falls back to deny-and-flag.

Evaluation. Synthetic patient data (no PHI) lacks the clinical variables real coverage criteria hinge on, so the 20-case suite scores only what's independently verifiable: deterministic auto-deny routing, rejection of clinically unrelated diagnosis/procedure pairs, citation groundedness (~0.90 mean), and 100% output-format compliance.

Stack: LangGraph · FAISS · sentence-transformers · Llama 3.1 8B on AWS Bedrock · Streamlit · SQLite.
