from dotenv import load_dotenv
load_dotenv()
 
import os
import json
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from google import genai

##############################################################################
 
api_key = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)
MODEL = "gemini-2.5-flash"
 
print("api key loading done")

############################################################################

# llm helper

def call_llm(prompt: str) -> str:
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text.strip()

#############################################################################

# 1. state typedict

class SupportState(TypedDict):
    user_input: str
    cleaned_input: str
    category: str
    draft_response: str
    eval_passed: bool
    eval_feedback: str
    final_response: str
    trace: list[str]

################################################################################

# 2. nodes

def clean_input(state: SupportState) -> dict:
    """Sequential chain step 1: normalize the raw user request via LLM."""
    raw = state["user_input"].strip()
 
    if raw == "":
        print("[NODE] clean_input     -> empty input, skipping LLM call")
        return {
            "cleaned_input": "",
            "trace": state["trace"] + ["clean_input"],
        }
 
    prompt = (
        "Rewrite the following support request as a single clear, "
        "grammatically correct sentence. DO NOT answer it, just clean it up. "
        "Only output the rewritten sentence, nothing else.\n\n"
        f"Request: {raw}"
    )
    cleaned = call_llm(prompt)
    print(f"[NODE] clean_input     -> cleaned_input={cleaned!r}")
    return {
        "cleaned_input": cleaned,
        "trace": state["trace"] + ["clean_input"],
    }



def route(state: SupportState) -> dict:
    """
    Rule-based routing -- no LLM call.
    Just records the category; the actual branch decision happens in the
    conditional edge function (route_decision) below.
    """
    text = state["cleaned_input"].lower()
 
    if text == "":
        category = "fallback"
    elif any(w in text for w in ["password", "login", "log in", "account access", "locked out"]):
        category = "password"
    elif any(w in text for w in ["refund", "charge", "bill", "invoice", "payment"]):
        category = "billing"
    elif any(w in text for w in ["order", "shipping", "delivery", "package", "delayed"]):
        category = "orders"
    else:
        category = "general"
 
    print(f"[NODE] route           -> category={category!r}")
    return {
        "category": category,
        "trace": state["trace"] + ["route"],
    }


def route_decision(state: SupportState) -> str:
    """Conditional edge function: reads state, returns next node name."""
    if state["category"] == "fallback":
        return "fallback"
    return "draft_response"


def fallback(state: SupportState) -> dict:
    """Safe canned response for empty/unusable input. No LLM call."""
    print("[NODE] fallback        -> returning safe default response")
    return {
        "final_response": "I didn't receive a question. Could you tell me what you need help with?",
        "trace": state["trace"] + ["fallback"],
    }


def draft_response(state: SupportState) -> dict:
    prompt = (
        f"You are a support assistant. The request category is '{state['category']}'.\n"
        f"Request: {state['cleaned_input']}\n\n"
        "Write a short, clear, helpful support response (2-4 sentences)."
    )
    draft = call_llm(prompt)
    print(f"[NODE] draft_response  -> draft={draft!r}")
    return {
        "draft_response": draft,
        "trace": state["trace"] + ["draft_response"],
    }


def evaluate(state: SupportState) -> dict:
    """Evaluator-optimizer, step 1: check draft against a tiny rubric."""
    prompt = (
        "Evaluate the following support response against two criteria: "
        "CLARITY (is it easy to understand?) and COMPLETENESS (does it fully "
        "address the request?).\n\n"
        f"Request: {state['cleaned_input']}\n"
        f"Response: {state['draft_response']}\n\n"
        "Reply with ONLY a JSON object, no markdown, in this exact format:\n"
        '{"passed": true or false, "feedback": "short explanation"}'
    )
    raw = call_llm(prompt)
    raw = raw.replace("```json", "").replace("```", "").strip()
 
    try:
        verdict = json.loads(raw)
        passed = bool(verdict.get("passed", True))
        feedback = str(verdict.get("feedback", ""))
    except (json.JSONDecodeError, AttributeError):
        # Safe fallback if the model doesn't return clean JSON -- don't crash the graph.
        passed = True
        feedback = "Could not parse evaluator output; passing draft through as-is."
 
    print(f"[NODE] evaluate        -> passed={passed}, feedback={feedback!r}")
    return {
        "eval_passed": passed,
        "eval_feedback": feedback,
        "trace": state["trace"] + ["evaluate"],
    }


def eval_decision(state: SupportState) -> str:
    """Conditional edge function: pass -> finalize, fail -> optimize once."""
    return "finalize" if state["eval_passed"] else "optimize"


def optimize(state: SupportState) -> dict:
    """Evaluator-optimizer, step 2: revise the draft once using feedback."""
    prompt = (
        f"Revise the following support response to fix this issue: "
        f"{state['eval_feedback']}\n\n"
        f"Original request: {state['cleaned_input']}\n"
        f"Original response: {state['draft_response']}\n\n"
        "Write only the improved response, 2-4 sentences."
    )
    revised = call_llm(prompt)
    print(f"[NODE] optimize        -> revised={revised!r}")
    return {
        "draft_response": revised,
        "trace": state["trace"] + ["optimize"],
    }



def finalize(state: SupportState) -> dict:
    final = state.get("final_response") or state["draft_response"]
    print(f"[NODE] finalize        -> final_response={final!r}")
    return {
        "final_response": final,
        "trace": state["trace"] + ["finalize"],
    }


################################################################################

# 3. graph builder

def build_graph():
    builder = StateGraph(SupportState)
 
    builder.add_node("clean_input", clean_input)
    builder.add_node("route", route)
    builder.add_node("fallback", fallback)
    builder.add_node("draft_response", draft_response)
    builder.add_node("evaluate", evaluate)
    builder.add_node("optimize", optimize)
    builder.add_node("finalize", finalize)
 
    builder.add_edge(START, "clean_input")
    builder.add_edge("clean_input", "route")
 
    # Routing: conditional edge based on category
    builder.add_conditional_edges(
        "route",
        route_decision,
        {"fallback": "fallback", "draft_response": "draft_response"},
    )
 
    builder.add_edge("fallback", END)
    builder.add_edge("draft_response", "evaluate")
 
    # Evaluator-optimizer: conditional edge based on pass/fail
    builder.add_conditional_edges(
        "evaluate",
        eval_decision,
        {"finalize": "finalize", "optimize": "optimize"},
    )
 
    builder.add_edge("optimize", "finalize")
    builder.add_edge("finalize", END)
 
    return builder.compile()


 #############################################################################

# 4. main

if __name__ == "__main__":
    graph = build_graph()
 
    test_inputs = [
        "hii my passwrd isnt working can u help",
        "",
        "i want a refund for my last order it never showed up",
    ]
 
    for raw_input in test_inputs:
        initial_state: SupportState = {
            "user_input": raw_input,
            "cleaned_input": "",
            "category": "",
            "draft_response": "",
            "eval_passed": False,
            "eval_feedback": "",
            "final_response": "",
            "trace": [],
        }
 
        print("=" * 60)
        print(f"ACT II -- running graph for input: {raw_input!r}")
        print("=" * 60)
 
        result = graph.invoke(initial_state)
 
        print("-" * 60)
        print("FINAL STATE:")
        for k, v in result.items():
            print(f"  {k}: {v}")
        print("=" * 60 + "\n")
 