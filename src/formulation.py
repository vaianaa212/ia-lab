from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Optional

@dataclass(frozen=True)
class Formulation:
    name: str
    state_schema: Dict[str, str]
    action_schema: Dict[str, str]
    transition_desc: str
    goal_test_desc: str
    cost_desc: str
    hard_constraints: List[str]
    soft_constraints: List[str]
    examples: Dict[str, Any]

def to_export_dict(f: Formulation) -> Dict[str, Any]:
    return {
        "case_name": f.name,
        "state_schema": f.state_schema,
        "action_schema": f.action_schema,
        "transition": f.transition_desc,
        "goal_test": f.goal_test_desc,
        "cost_function": {"description": f.cost_desc},
        "constraints": {"hard": f.hard_constraints, "soft": f.soft_constraints},
        "examples": f.examples,
    }
