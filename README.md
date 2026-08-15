# Support Copilot — LangGraph POC

> Built for the **Sprints Advanced Agentic AI** course, as a freelance-style
> proof-of-concept task demonstrating LangGraph workflow patterns.

A lightweight proof-of-concept showing how LangGraph can orchestrate a real
multi-step support workflow — not just a single prompt-in, prompt-out call.

The project is built in three acts, each adding a workflow pattern on top of
the last, ending in one graph that demonstrates **five** patterns working
together: sequential chaining, routing, evaluator-optimizer,
parallelization, and orchestrator-worker.

---

## 1. Setup

### Requirements

- Python 3.11+
- A Gemini API key

### Windows

```bash
# from the project folder
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### macOS

```bash
# from the project folder
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

---

## 2. Running each act

Each act is a standalone, runnable script, located in `src/`. Run them from
the **project root** (not from inside `src/`) so the `.env` file is found
correctly:
 
```bash
python src/act1_graph.py   # no API key needed — pure graph mechanics
python src/act2_graph.py   # needs GEMINI_API_KEY — sequential chain + routing + evaluator-optimizer
python src/act3_graph.py   # needs GEMINI_API_KEY — adds parallelization + orchestrator-worker
```

Each script prints a log line per node as it executes (`[NODE] name -> ...`),
followed by the full final state, so the graph's behavior is visible without
needing a debugger.

> **Note on API quota:** the free Gemini tier is limited to 5
> requests/minute. `act3_graph.py`'s complex-question path alone uses ~6
> calls (clean → orchestrator → 2-3 workers in parallel → synthesize →
> evaluate). If you hit a `429 RESOURCE_EXHAUSTED` error, wait ~60 seconds
> and re-run.

---

## 3. The workflow story

### Act I : From Empty Folder to First Graph Run

Proves the LangGraph mechanics work before any LLM is involved. A tiny
3-node graph (`receive_input → process → finalize`) passes a shared state
object through a fixed sequence of edges, logging every transition.

### Act II : A Chatbot That Thinks in Steps

Introduces the LLM. A user request is:
1. **Cleaned up** via a short sequential chain (fixes typos/grammar into one clear sentence)
2. **Routed** to a category (`password`, `billing`, `orders`, `general`, or `fallback`) — this is a cheap rule-based/keyword decision, not an LLM call, to avoid burning an extra request on something regex can do
3. **Drafted** into a response by the LLM
4. **Evaluated** against a tiny rubric (clarity + completeness); if it fails, it's **revised once** (evaluator-optimizer pattern) — the graph's edge structure itself guarantees only one revision, not a loop

Empty/garbage input short-circuits straight to a safe fallback response,
skipping the LLM entirely.

### Act III : Scaling the Workflow (Without Scaling the Product)
Adds two more patterns on top of Act II's graph:

- **Parallelization** — for simple requests, a full draft response and a
  short TL;DR summary are generated **at the same time**, then merged into
  one message.
- **Orchestrator-worker** — for complex, multi-part requests (detected by a
  lightweight heuristic — no LLM call needed), an orchestrator node breaks
  the request into 2-3 sub-tasks, a dynamic number of worker nodes solve
  each one in parallel, and a synthesize step combines their answers into
  one coherent response.

Both new paths feed into the **same** evaluator-optimizer loop from Act II
— everything converges into one coherent conversation, not three
disconnected demos.

---

## 4. Design decisions worth noting

- **Routing is rule-based, not LLM-based** — keeps the graph lean and avoids
  redundant API calls (a stated non-functional requirement).
- **The evaluator returns strict JSON**, parsed with a `try/except` fallback
  — if the LLM doesn't format its response correctly, the graph still runs
  instead of crashing.
- **The one-revision guarantee is structural, not logical** — the `optimize`
  node has exactly one way in (from a failed evaluation) and one way out
  (straight to `finalize`), so there's no way for the graph to loop
  indefinitely even without an explicit revision counter.
- **`trace` and `worker_outputs` use LangGraph reducers**
  (`Annotated[list, operator.add]`) in Act III, since parallel nodes write
  to them in the same step and need their contributions concatenated rather
  than overwritten.

---

## 5. Files

| File | Purpose |
|---|---|
| `act1_graph.py` | Bare-bones graph: nodes, edges, shared state |
| `act2_graph.py` | Adds LLM chain, routing, evaluator-optimizer |
| `act3_graph.py` | Adds parallelization, orchestrator-worker |
| `test_prompts.md` | Sample prompts and which workflow path each triggers |
| `requirements.txt` | Python dependencies |
| `.env` | Your local Gemini API key (not committed) |
