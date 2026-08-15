# Test Prompts

Prompts used to demo the Support Copilot, grouped by which workflow path
each one is designed to trigger. Use these when capturing demo logs/screenshots.

---

## Act II demo prompts

### 1. Sequential chain + routing + evaluator-optimizer (pass on first try)
```
hii my passwrd isnt working can u help
```
**Expected path:** `clean_input → route (category=password) → draft_response → evaluate (passed=True) → finalize`

### 2. Routing to a different category
```
i want a refund for my last order it never showed up
```
**Expected path:** `clean_input → route (category=billing) → draft_response → evaluate (passed=True) → finalize`

### 3. Fallback (empty/no input)
```

```
**Expected path:** `clean_input (skipped, empty) → route (category=fallback) → fallback → END`
*(No LLM calls at all — proves the robustness/safe-fallback requirement.)*

### 4. Evaluator-optimizer loop — deliberately triggers a REVISION
```
help
```
A single vague word like this is likely to produce a draft response that
fails the "completeness" rubric check on first pass, triggering the
`optimize` node. Run this a few times if needed — LLM outputs vary, so it
won't fail 100% of the time.

**Expected path (when it fails):** `clean_input → route → draft_response → evaluate (passed=False) → optimize → finalize`

---

## Act III demo prompts

### 5. Parallelization (simple request → draft + summary in parallel)
```
how do i change my email address on my account
```
**Expected path:** `clean_input → route (is_complex=False) → [draft_response, draft_summary] (parallel) → merge → evaluate → finalize`

### 6. Orchestrator-worker (complex, multi-part request)
```
why is my order delayed, will i get a refund, and how do i update my shipping address for next time
```
**Expected path:** `clean_input → route (is_complex=True) → orchestrator (breaks into 2-3 subtasks) → worker (runs once per subtask, in parallel) → synthesize → evaluate → finalize`

### 7. Another orchestrator-worker example (for variety in the demo)
```
my payment failed twice, I want to know why, whether I was charged anyway, and how to update my card
```
**Expected path:** same shape as #6 — 3 distinct sub-tasks (payment failure reason, charge status, card update).