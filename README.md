# Prior Auth Agent

This RAG-based agent with an eval harness evaluates whether a requested procedure meets Medicare coverage criteria and produces an auditable rationale tied to retrieved CMS policy language. 

## Architecture

Requests move through a bounded workflow implemented as a [LangGraph](https://github.com/langchain-ai/langgraph) state graph. After initial retrieval, an evidence agent can make a limited number of read-only tool calls to refine its policy search or inspect a specific patient-record section. 

## Policy Grounding

Three CMS Local Coverage Determinations (LCDs) included in `docs/cms_policies/` are:

1. Parsed and split into chunks.
2. Embedded and indexed with FAISS at startup.
3. Retrieved using the requested procedure as the search query.
4. Passed to the model alongside the relevant patient record.

The decision prompt requires the model to identify the specific policy language supporting its conclusion. This provides a traceable relationship between the generated decision and the retrieved source material and allows citations to be evaluated independently for groundedness.

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
* **Semantic rubric:** A separate LLM-as-a-judge scores citation entailment and decision support using only the retrieved policy evidence. A well-grounded `PENDED` response can pass when manual review is appropriate.
* **Operational metrics:** Mean, p50, and p95 latency; mean and maximum tool calls; manual-review rate; errors; and a pass/fail check against a 120-second / five-tool-call ceiling.

Cases in which the diagnosis and procedure are clinically aligned—for example, low back pain with lumbar MRI—are executed and logged but are not assigned an accuracy score. The available synthetic records do not contain enough information to establish a defensible coverage ground truth for these cases. They are therefore intended for qualitative review of retrieval and reasoning behavior.

### Failure Handling

Evaluation identified a generation reliability issue with Llama 3.1 8B: in some cases, the model entered a repetition loop and exhausted its generation budget before producing the required policy citation.

`graph.py` includes explicit handling for this failure mode. It detects malformed or repetitive output and retries generation with an increased temperature. If repeated attempts still fail to produce a valid cited response, the request returns a deny-and-flag-for-review result rather than treating an incomplete generation as a valid decision.

Run the evaluation suite with:

```bash
PYTHONPATH=. python3 eval/run_eval.py
```

To run deterministic and operational checks without the additional judge call for each non-auto-deny case:

```bash
PYTHONPATH=. python3 eval/run_eval.py --skip-semantic-judge
```
