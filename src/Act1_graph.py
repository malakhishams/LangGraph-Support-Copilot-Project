from typing import TypedDict
from langgraph.graph import StateGraph, START, END

#############################################################################

# 1. state typedict

class SupportState(TypedDict):
    user_input: str
    processed_input: str
    trace: list[str] 

################################################################################

# 2. nodes

def receive_input(state: SupportState) -> dict:
    print(f"[NODE] receive_input   -> got user_input={state['user_input']!r}")
    return {
        "trace": state["trace"] + ["receive_input"]
    }

def process(state: SupportState) -> dict:
    cleaned = state["user_input"].strip().lower()
    print(f"[NODE] process         -> processed_input={cleaned!r}")
    return {
        "processed_input": cleaned,
        "trace": state["trace"] + ["process"]
    }

def finalize(state: SupportState) -> dict:
    print(f"[NODE] finalize        -> final trace={state['trace'] + ['finalize']}")
    return {
        "trace": state["trace"] + ["finalize"]
    }


###############################################################################

# 3. graph builder

def build_graph():
    builder = StateGraph(SupportState)
 
    builder.add_node("receive_input", receive_input)
    builder.add_node("process", process)
    builder.add_node("finalize", finalize)
 
    # Straight line: START -> receive_input -> process -> finalize -> END
    builder.add_edge(START, "receive_input")
    builder.add_edge("receive_input", "process")
    builder.add_edge("process", "finalize")
    builder.add_edge("finalize", END)
 
    return builder.compile()

################################################################################

# 4. main

graph = build_graph()
 
initial_state: SupportState = {
        "user_input": "  Hello, My Password Is Not Working!  ",
        "processed_input": "",
        "trace": [],
    }
 
print("=" * 60)
print("ACT I -- running skeleton graph")
print("=" * 60)

result = graph.invoke(initial_state)

print("=" * 60)
print("FINAL STATE:")
for k, v in result.items():
    print(f"  {k}: {v}")
print("=" * 60)
