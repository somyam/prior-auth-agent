# Prior Auth Agent

A policy-grounded prior authorization agent that evaluates whether a requested procedure meets Medicare coverage criteria and produces an auditable rationale tied to retrieved CMS policy language.

The system automates a common prior authorization workflow: matching a procedure request and patient record against applicable coverage criteria. Rather than relying on the model's internal knowledge of Medicare policy, it retrieves relevant policy text and requires each model-generated decision to include a supporting citation.

## Architecture

Requests move through a fixed workflow implemented as a [LangGraph](https://github.com/langchain-ai/langgraph) state graph. Each stage is represented as an isolated node, with explicit conditional routing for cases that should not proceed to model inference.

```text
                 ┌──────────────────┐
   diagnosis ──> │  patient lookup  │
   lookup        └────────┬─────────┘
                          │
              ┌───────────┴────────────┐
        patient found            patient not found
              │                         │
              ▼                         ▼
      policy retrieval             auto-deny
              │                         │
              ▼                         │
        model decision                  │
              │                         │
              └───────────┬─────────────┘
                          ▼
                     audit record
```

If no matching patient record is found, the request is denied and logged without invoking the model. This prevents inference against an unverified or unavailable patient record.

## Policy Grounding

The model is restricted to retrieved CMS policy text rather than its pretrained knowledge of Medicare coverage rules.

Three CMS Local Coverage Determinations (LCDs) in `docs/cms_policies/` are:

1. Parsed and split into chunks.
2. Embedded and indexed with FAISS at startup.
3. Retrieved using the requested procedure as the search query.
4. Passed to the model alongside the relevant patient record.

The decision prompt requires the model to identify the specific policy language supporting its conclusion. This provides a traceable relationship between the generated decision and the retrieved source material and allows citations to be evaluated independently for groundedness.

## Repository Structure

| File                 | Responsibility                                                            |
| -------------------- | ------------------------------------------------------------------------- |
| `graph.py`           | Defines the LangGraph workflow, conditional routing, and decision prompt  |
| `tools.py`           | Diagnosis-code lookup, patient-record lookup, and FAISS policy retrieval  |
| `index.py`           | Parses, chunks, and embeds CMS policy PDFs and caches the resulting index |
| `audit.py`           | Persists approvals, denials, and automatic denials to SQLite              |
| `app.py`             | Streamlit interface for submitting requests and reviewing the audit log   |
| `data/patients.csv`  | Synthetic patient records; contains no real PHI                           |
| `docs/cms_policies/` | CMS Local Coverage Determination PDFs used for retrieval                  |
| `eval/`              | Evaluation cases and evaluation harness                                   |

## Running Locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add AWS Bedrock credentials
streamlit run app.py
```

On the first run, the application builds a FAISS index from the policy PDFs and stores it in `.cache/`. Subsequent runs load the cached index.

## Evaluation

The synthetic patient records do not contain all of the clinical variables required to determine Medicare coverage for every procedure, such as symptom duration, prior conservative treatment, or screening intervals. As a result, the evaluation does **not** treat model decisions as clinically validated ground truth.

Instead, `eval/run_eval.py` evaluates properties that can be verified directly against a 20-case test set in `eval/cases.jsonl`:

* **Auto-deny routing — 4/4:** Requests containing an unknown patient ID must be denied without invoking the model.
* **Diagnosis/procedure mismatch — 8/8:** Requests with an unrelated diagnosis and procedure must be denied when the retrieved policy provides no basis for coverage.
* **Citation groundedness — ~0.90 mean:** Measures the extent to which cited policy text can be traced verbatim to the retrieved context supplied to the model. This is used as a proxy for unsupported or hallucinated citations.
* **Output-format compliance — 100%:** Measures whether responses conform to the required `DECISION / REASONING / POLICY CITATION` structure.

Cases in which the diagnosis and procedure are clinically aligned—for example, low back pain with lumbar MRI—are executed and logged but are not assigned an accuracy score. The available synthetic records do not contain enough information to establish a defensible coverage ground truth for these cases. They are therefore intended for qualitative review of retrieval and reasoning behavior.

### Failure Handling

Evaluation identified a generation reliability issue with Llama 3.1 8B: in some cases, the model entered a repetition loop and exhausted its generation budget before producing the required policy citation.

`graph.py` includes explicit handling for this failure mode. It detects malformed or repetitive output and retries generation with an increased temperature. If repeated attempts still fail to produce a valid cited response, the request returns a deny-and-flag-for-review result rather than treating an incomplete generation as a valid decision.

Run the evaluation suite with:

```bash
PYTHONPATH=. python3 eval/run_eval.py
```

## Scope

This repository is a prototype for evaluating policy-grounded prior authorization workflows. It uses synthetic patient data and a limited set of CMS policies and is not intended to make production clinical or coverage determinations.
