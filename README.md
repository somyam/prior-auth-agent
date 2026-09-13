# Prior Auth Agent

A policy-grounded prior authorization agent that evaluates whether a requested procedure meets Medicare coverage criteria and produces an auditable rationale tied to retrieved CMS policy language.

The system automates a common prior authorization workflow: matching a procedure request and patient record against applicable coverage criteria. Rather than relying on the model's internal knowledge of Medicare policy, it retrieves relevant policy text and requires each model-generated decision to include a supporting citation.

## Architecture

Requests move through a bounded workflow implemented as a [LangGraph](https://github.com/langchain-ai/langgraph) state graph. After initial retrieval, an evidence agent can make a limited number of read-only, whitelisted tool calls to refine its policy search or inspect a specific patient-record section. Deterministic validation owns the final routing.

```text
                 ┌──────────────────┐
   diagnosis ──> │  patient lookup  │
   lookup        └────────┬─────────┘
                          │
              ┌───────────┴────────────┐
        patient found            patient not found
              │                         │
              ▼                         ▼
      initial policy retrieval     auto-deny
              │                         │
              ▼                         │
  bounded evidence-agent loop            │
  (policy search / patient inspection)   │
              │                         │
              ▼                         │
     citation + criteria validator        │
              │                         │
              └───────────┬─────────────┘
                          ▼
                     audit record
```

If no matching patient record is found, the request is denied and logged without invoking the model. This prevents inference against an unverified or unavailable patient record.

The agent has only two read-only tools: policy search and retrieval of one of three patient-record sections (conditions, medications, or procedures). It is capped at four calls. It must finalize with a structured criterion-by-criterion assessment. An approval requires every cited criterion to be documented as met; a denial requires a documented unmet criterion; missing or ambiguous evidence becomes `PENDED` for manual review. The verifier checks that every policy quote appears in the retrieved evidence before a decision is accepted.

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
| `graph.py`           | Defines the bounded LangGraph evidence-agent loop, routing, and validator |
| `tools.py`           | Diagnosis-code lookup, patient-record lookup, and FAISS policy retrieval  |
| `index.py`           | Parses, chunks, and embeds CMS policy PDFs and caches the resulting index |
| `audit.py`           | Persists decisions plus the agent trace, assessment, and policy evidence  |
| `app.py`             | Streamlit interface for submitting requests and reviewing the audit log   |
| `data/patients.csv`  | Synthetic patient records; contains no real PHI                           |
| `docs/cms_policies/` | CMS Local Coverage Determination PDFs used for retrieval                  |
| `eval/`              | Evaluation cases and evaluation harness                                   |

## Running Locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

By default, the app calls a local [Ollama generate endpoint](https://docs.ollama.com/api/generate) at `http://localhost:11434` with `llama3.1:8b`. Install Ollama, pull that model (or select another capable local model), and start its local service before launching the app:

```bash
ollama pull llama3.1:8b
```

Optional environment variables:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
OLLAMA_URL=http://localhost:11434/api/generate
```

Bedrock remains available as an opt-in provider for deployments that need it:

```env
LLM_PROVIDER=bedrock
AWS_REGION=us-east-2
AWS_ACCESS_KEY=your_access_key
AWS_SECRET_KEY=your_secret_key
```

On the first run, the application downloads the embedding model, builds a FAISS index from the policy PDFs, and stores it in `.cache/`. Subsequent runs load the cached index.

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
