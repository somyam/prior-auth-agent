# Prior Auth Agent

A small agent that answers one question — should this procedure be approved
under Medicare policy? — and shows exactly how it got there.

Manual prior-auth review means a person reading a procedure request
alongside a lengthy coverage policy and cross-checking one against the
other. This project automates that cross-check: retrieve the relevant
policy language, hand it to the model along with the patient's record, and
require the model to point at the specific clause its decision rests on.
No clause, no decision.

## Design

The request moves through a fixed sequence of steps, modeled as a
[LangGraph](https://github.com/langchain-ai/langgraph) graph rather than a
single long function — each step is an isolated node, and the graph branches
on real conditions instead of silently falling through:

```
                 ┌──────────────────┐
   diagnosis ──> │  patient lookup   │
   lookup        └────────┬──────────┘
                           │
              ┌────────────┴────────────┐
        patient on file           no matching patient
              │                          │
              ▼                          ▼
      policy retrieval             auto-deny
              │                          │
              ▼                          │
        model decision                   │
              │                          │
              └───────────┬──────────────┘
                           ▼
                      audit record
```

A request for an unrecognized patient never reaches the model — it's
denied and logged automatically, which also means the model is never asked
to reason about a patient it can't actually verify.

## Grounding the decision

The model doesn't get to reference what it remembers about Medicare policy.
The three CMS Local Coverage Determinations in `docs/cms_policies/` are
split into chunks, embedded, and indexed with FAISS at startup. Each
request embeds the procedure being requested, pulls the closest matching
chunks, and only that retrieved text is placed in front of the model —
along with an explicit instruction to quote the clause it used. That
citation is what makes the decision checkable after the fact rather than
just plausible-sounding.

## What's in each file

| File | Responsibility |
|---|---|
| `graph.py` | Defines the state graph: nodes, the patient-found branch, the decision prompt |
| `tools.py` | Diagnosis code lookup, patient record lookup, FAISS policy search |
| `index.py` | Chunks and embeds the policy PDFs, caches the index to `.cache/` |
| `audit.py` | Writes every decision — approvals, denials, and auto-denials — to SQLite |
| `app.py` | Streamlit form for submitting a request and browsing the audit log |
| `data/patients.csv` | Synthetic patient records (no real PHI) |
| `docs/cms_policies/` | The three CMS LCD PDFs the agent retrieves from |
| `eval/` | Eval harness — see [Evaluation](#evaluation) |

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your AWS Bedrock credentials
streamlit run app.py
```

The first run builds the FAISS index from the policy PDFs and caches it;
later runs load straight from `.cache/`.

## Evaluation

The synthetic patient records don't carry the clinical detail (symptom
duration, documented conservative-therapy trials, screening intervals)
that the actual CMS criteria hinge on, so there's no ground truth for
"was this specific decision clinically correct." `eval/run_eval.py`
measures what's actually verifiable instead, against a 20-case set in
`eval/cases.jsonl`:

- **Auto-deny routing** — an unknown patient ID must be denied without
  ever reaching the model. 4/4.
- **Diagnosis/procedure mismatch** — a request where the diagnosis has
  nothing to do with the procedure (a GI diagnosis requesting a knee
  replacement, say) must be denied regardless of patient history, since
  no policy would support it. 8/8.
- **Citation groundedness** — how much of the model's cited policy text
  is actually traceable, word-for-word, to the chunks it was given, as a
  proxy for hallucinated citations. ~0.90 average.
- **Format compliance** — does the response parse into
  DECISION/REASONING/POLICY CITATION. 100%.

A remaining bucket of "aligned" cases (diagnosis and procedure in the
same clinical category, e.g. low back pain + lumbar MRI) has no
assignable ground truth, so those run and get logged but aren't scored
for accuracy — useful for spot-checking reasoning quality, not a
pass/fail number.

Running the eval is also what surfaced a real reliability bug: Llama
3.1 8B would occasionally fall into a repetition loop at low temperature
and burn its whole generation budget without ever producing a policy
citation. `graph.py` now detects that pattern and retries with escalating
temperature, falling back to an explicit deny-and-flag-for-review result
if it still can't produce a well-formed, cited decision.

```bash
PYTHONPATH=. python3 eval/run_eval.py
```

## Known gaps

- The model returns a binary decision with no confidence signal — there's
  no way to distinguish a clear-cut case from a borderline one it decided
  anyway.
- Every audit record lives in one local SQLite file; there's no
  multi-writer story if this ran as more than a single-user demo.

Patient data is synthetic throughout. No real PHI is used at any stage.
