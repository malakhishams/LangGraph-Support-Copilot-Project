import os
import json
import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from google import genai
from dotenv import load_dotenv
load_dotenv()
 

###############################################################################


api_key = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)
MODEL = "gemini-2.5-flash"
 
print("api key loading done")

###############################################################################

# llm helper

def call_llm(prompt: str) -> str:
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text.strip()

###############################################################################

# 1. state typedict

class SupportState(TypedDict):
    user_input: str
    cleaned_input: str
    category: str
    is_complex: bool
 
    # simple / parallel path
    draft_response: str
    summary: str
 
    # complex / orchestrator-worker path
    subtasks: list[str]
    worker_outputs: Annotated[list[str], operator.add]
 
    # shared downstream
    eval_passed: bool
    eval_feedback: str
    final_response: str
    trace: Annotated[list[str], operator.add]


###############################################################################

# 2. nodes

def clean_input(state: SupportState) -> dict:
    raw = state["user_input"].strip()
    if raw == "":
        print("[NODE] clean_input     -> empty input, skipping LLM call")
        return {"cleaned_input": "", "trace": ["clean_input"]}
 
    prompt = (
        "Rewrite the following support request as a single clear, "
        "grammatically correct sentence. Do not answer it, just clean it up. "
        "Only output the rewritten sentence, nothing else.\n\n"
        f"Request: {raw}"
    )
    cleaned = call_llm(prompt)
    print(f"[NODE] clean_input     -> cleaned_input={cleaned!r}")
    return {"cleaned_input": cleaned, "trace": ["clean_input"]}


 
def is_complex_request(text: str) -> bool:
    """
    Lightweight heuristic, no LLM call: a request is "complex" if it looks
    like it's asking multiple things at once.
    """
    if text == "":
        return False
    signal_count = text.count("?") + text.lower().count(" and ") + text.count(",")
    return signal_count >= 2 or len(text.split()) > 25



def route(state: SupportState) -> dict:
    text = state["cleaned_input"].lower()
 
    if text == "":
        category = "fallback"
    elif any(w in text for w in ["password", "login", "log in", "locked out"]):
        category = "password"
    elif any(w in text for w in ["refund", "charge", "bill", "invoice", "payment"]):
        category = "billing"
    elif any(w in text for w in ["order", "shipping", "delivery", "package", "delayed"]):
        category = "orders"
    else:
        category = "general"
 
    complex_flag = is_complex_request(state["cleaned_input"])
    print(f"[NODE] route           -> category={category!r}, is_complex={complex_flag}")
    return {
        "category": category,
        "is_complex": complex_flag,
        "trace": ["route"],
    }



def route_decision(state: SupportState):
    """
    Conditional edge after route(). Returns either a single node name or a
    LIST of node names -- returning a list is how LangGraph fans out to
    multiple nodes in parallel (the parallel branch for simple requests).
    """
    if state["category"] == "fallback":
        return "fallback"
    if state["is_complex"]:
        return "orchestrator"
    return ["draft_response", "draft_summary"]



def fallback(state: SupportState) -> dict:
    print("[NODE] fallback        -> returning safe default response")
    return {
        "final_response": "I didn't receive a question. Could you tell me what you need help with?",
        "trace": ["fallback"],
    }


def evaluate(state: SupportState) -> dict:
    prompt = (
        "Evaluate the following support response against two criteria: "
        "CLARITY (is it easy to understand?) and COMPLETENESS (does it fully "
        "address the request?).\n\n"
        f"Request: {state['cleaned_input']}\n"
        f"Response: {state['draft_response']}\n\n"
        "Reply with ONLY a JSON object, no markdown, in this exact format:\n"
        '{"passed": true or false, "feedback": "short explanation"}'
    )
    raw = call_llm(prompt).replace("```json", "").replace("```", "").strip()
 
    try:
        verdict = json.loads(raw)
        passed = bool(verdict.get("passed", True))
        feedback = str(verdict.get("feedback", ""))
    except (json.JSONDecodeError, AttributeError):
        passed = True
        feedback = "Could not parse evaluator output; passing draft through as-is."
 
    print(f"[NODE] evaluate        -> passed={passed}, feedback={feedback!r}")
    return {"eval_passed": passed, "eval_feedback": feedback, "trace": ["evaluate"]}




def eval_decision(state: SupportState) -> str:
    return "finalize" if state["eval_passed"] else "optimize"

 
 
def optimize(state: SupportState) -> dict:
    prompt = (
        f"Revise the following support response to fix this issue: "
        f"{state['eval_feedback']}\n\n"
        f"Original request: {state['cleaned_input']}\n"
        f"Original response: {state['draft_response']}\n\n"
        "Write only the improved response, 2-4 sentences."
    )
    revised = call_llm(prompt)
    print(f"[NODE] optimize        -> revised={revised!r}")
    return {"draft_response": revised, "trace": ["optimize"]}


 
def finalize(state: SupportState) -> dict:
    final = state.get("final_response") or state["draft_response"]
    print(f"[NODE] finalize        -> final_response={final!r}")
    return {"final_response": final, "trace": ["finalize"]}



def draft_response(state: SupportState) -> dict:
    prompt = (
        f"You are a support assistant. The request category is '{state['category']}'.\n"
        f"Request: {state['cleaned_input']}\n\n"
        "Write a short, clear, helpful support response (2-4 sentences)."
    )
    draft = call_llm(prompt)
    print(f"[NODE] draft_response  -> draft={draft!r}")
    return {"draft_response": draft, "trace": ["draft_response"]}



def draft_summary(state: SupportState) -> dict:
    prompt = (
        f"Summarize the following support request in ONE short sentence "
        f"(a TL;DR, not an answer):\n\n{state['cleaned_input']}"
    )
    summary = call_llm(prompt)
    print(f"[NODE] draft_summary   -> summary={summary!r}")
    return {"summary": summary, "trace": ["draft_summary"]}



def merge(state: SupportState) -> dict:
    """
    Runs only after BOTH draft_response and draft_summary complete --
    LangGraph waits for all incoming edges into a node before running it.
    Plain string merge, no LLM call needed here.
    """
    merged = f"{state['draft_response']}\n\nTL;DR: {state['summary']}"
    print(f"[NODE] merge           -> merged draft ready")
    return {"draft_response": merged, "trace": ["merge"]}



 
def orchestrator(state: SupportState) -> dict:
    prompt = (
        "Break the following support request into 2 or 3 short, distinct "
        "sub-tasks that together fully answer it. Reply with ONLY a JSON "
        'list of strings, e.g. ["sub-task 1", "sub-task 2"], nothing else.\n\n'
        f"Request: {state['cleaned_input']}"
    )
    raw = call_llm(prompt).replace("```json", "").replace("```", "").strip()
 
    try:
        subtasks = json.loads(raw)
        if not isinstance(subtasks, list) or not subtasks:
            raise ValueError("empty or malformed subtask list")
    except (json.JSONDecodeError, ValueError):
        # Safe fallback: treat the whole request as a single subtask.
        subtasks = [state["cleaned_input"]]
 
    print(f"[NODE] orchestrator    -> subtasks={subtasks}")
    return {"subtasks": subtasks, "trace": ["orchestrator"]}



 
def assign_workers(state: SupportState):
    """
    Conditional edge after orchestrator(). Returns a list of Send objects --
    one per subtask. LangGraph dynamically creates that many parallel runs
    of the "worker" node, each with its own state payload.
    """
    return [
        Send("worker", {"subtask": task, "cleaned_input": state["cleaned_input"]})
        for task in state["subtasks"]
    ]



def worker(state: dict) -> dict:
    """
    Each worker only receives what assign_workers sent it: subtask + cleaned_input.
    Its output is appended to worker_outputs (reducer field), never overwritten.
    """
    subtask = state["subtask"]
    prompt = (
        f"Regarding this support request: {state['cleaned_input']}\n"
        f"Answer only this specific part: {subtask}\n"
        "Be concise: 1-2 sentences."
    )
    output = call_llm(prompt)
    print(f"[NODE] worker          -> subtask={subtask!r} -> output={output!r}")
    return {"worker_outputs": [output], "trace": ["worker"]}



def synthesize(state: SupportState) -> dict:
    """
    Runs once all workers finish. Combines their outputs into one
    coherent answer -- this DOES need an LLM call, since stitching
    independent answers into something coherent is exactly what a
    plain string join can't do well.
    """
    joined = "\n".join(f"- {o}" for o in state["worker_outputs"])
    prompt = (
        f"Combine these partial answers into ONE coherent, well-organized "
        f"support response to: {state['cleaned_input']}\n\n"
        f"Partial answers:\n{joined}"
    )
    combined = call_llm(prompt)
    print(f"[NODE] synthesize      -> combined draft ready")
    return {"draft_response": combined, "trace": ["synthesize"]}


##################################################################################################################

# 3. graph builder


def build_graph():
    builder = StateGraph(SupportState)
 
    builder.add_node("clean_input", clean_input)
    builder.add_node("route", route)
    builder.add_node("fallback", fallback)
    builder.add_node("draft_response", draft_response)
    builder.add_node("draft_summary", draft_summary)
    builder.add_node("merge", merge)
    builder.add_node("orchestrator", orchestrator)
    builder.add_node("worker", worker)
    builder.add_node("synthesize", synthesize)
    builder.add_node("evaluate", evaluate)
    builder.add_node("optimize", optimize)
    builder.add_node("finalize", finalize)
 
    builder.add_edge(START, "clean_input")
    builder.add_edge("clean_input", "route")
 
    # Three-way routing: fallback | parallel fan-out | orchestrator
    builder.add_conditional_edges("route", route_decision)
 
    builder.add_edge("fallback", END)
 
    # Parallel branch converges at merge
    builder.add_edge("draft_response", "merge")
    builder.add_edge("draft_summary", "merge")
    builder.add_edge("merge", "evaluate")
 
    # Orchestrator-worker branch converges at synthesize
    builder.add_conditional_edges("orchestrator", assign_workers)
    builder.add_edge("worker", "synthesize")
    builder.add_edge("synthesize", "evaluate")
 
    # Shared evaluator-optimizer loop
    builder.add_conditional_edges(
        "evaluate", eval_decision, {"finalize": "finalize", "optimize": "optimize"}
    )
    builder.add_edge("optimize", "finalize")
    builder.add_edge("finalize", END)
 
    return builder.compile()


########################################################################################
# 4. main


if __name__ == "__main__":
    graph = build_graph()
 
    test_inputs = [
        "why is my order delayed, will i get a refund, and how do i update my shipping address for next time"
    ]
 
    for raw_input in test_inputs:
        initial_state: SupportState = {
            "user_input": raw_input,
            "cleaned_input": "",
            "category": "",
            "is_complex": False,
            "draft_response": "",
            "summary": "",
            "subtasks": [],
            "worker_outputs": [],
            "eval_passed": False,
            "eval_feedback": "",
            "final_response": "",
            "trace": [],
        }
 
        print("=" * 60)
        print(f"ACT III -- running graph for input: {raw_input!r}")
        print("=" * 60)
 
        result = graph.invoke(initial_state)
 
        print("-" * 60)
        print("FINAL STATE:")
        for k, v in result.items():
            print(f"  {k}: {v}")
        print("=" * 60 + "\n")
 
