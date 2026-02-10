from typing import Dict, Any, List, Tuple
from .formulation import Formulation

def get_formulations() -> Dict[str, Formulation]:
    # Two representations for each case to support the "granularity trade-off" concept.
    turnos_compacto = Formulation(
        name="turnos_compacto",
        state_schema={
            "assigned": "dict{(day,shift): employee_id | None}",
            "workload": "dict{employee_id: turns_assigned_week}"
        },
        action_schema={
            "assign": "(day, shift, employee_id)"
        },
        transition_desc="Assign employee_id to (day,shift); increment workload; mark slot as filled.",
        goal_test_desc="All (day,shift) slots are filled AND hard constraints satisfied.",
        cost_desc="Sum of penalties: preference_violations*w1 + workload_imbalance*w2 + overtime*w3",
        hard_constraints=[
            "Each slot has exactly 1 employee",
            "No employee works two shifts in the same day",
            "Max turns per week per employee"
        ],
        soft_constraints=[
            "Avoid assigning on unavailable days (penalty)",
            "Balance workload across employees (penalty)"
        ],
        examples={
            "initial_state": {"assigned": "all None", "workload": "all 0"},
            "sample_action": {"assign": ("Mon", "AM", "E1")}
        }
    )

    turnos_enriquecido = Formulation(
        name="turnos_enriquecido",
        state_schema={
            "assigned": "dict{(day,shift): employee_id | None}",
            "workload": "dict{employee_id: turns_assigned_week}",
            "last_shift": "dict{employee_id: last_shift_type}"
        },
        action_schema={
            "assign": "(day, shift, employee_id)"
        },
        transition_desc="Assign employee; update workload and last_shift to enforce rest/sequence rules.",
        goal_test_desc="All slots filled; hard constraints satisfied incl. sequence/rest constraints.",
        cost_desc="Penalties as in compact + extra penalty for undesired sequences (e.g., PM->AM).",
        hard_constraints=[
            "Each slot has exactly 1 employee",
            "No employee works two shifts in the same day",
            "Max turns per week per employee",
            "Rest/sequence rules (e.g., no PM then next-day AM)"
        ],
        soft_constraints=[
            "Avoid unavailable days (penalty)",
            "Balance workload (penalty)",
            "Prefer stable schedules (penalty for changes)"
        ],
        examples={
            "initial_state": {"assigned": "all None", "workload": "all 0", "last_shift": "None"},
            "sample_action": {"assign": ("Mon", "PM", "E2")}
        }
    )

    picking_compacto = Formulation(
        name="picking_compacto",
        state_schema={
            "pos": "(x,y) current position in grid",
            "remaining": "set{locations to visit}"
        },
        action_schema={
            "move": "one-step move in {up,down,left,right} OR jump-to-location (if modeling as graph)"
        },
        transition_desc="Update pos; if pos is a target location, remove it from remaining.",
        goal_test_desc="remaining is empty.",
        cost_desc="Each move costs 1 (or distance); total cost is path length.",
        hard_constraints=[
            "Cannot move into blocked cells"
        ],
        soft_constraints=[
            "Prefer avoiding 'congested' aisles (penalty)"
        ],
        examples={
            "initial_state": {"pos": (0,0), "remaining": [(2,2),(4,1)]},
            "sample_action": {"move": "right"}
        }
    )

    picking_enriquecido = Formulation(
        name="picking_enriquecido",
        state_schema={
            "pos": "(x,y)",
            "remaining": "set{locations}",
            "time": "current time step (to model time windows or congestion)"
        },
        action_schema={
            "move": "{up,down,left,right}"
        },
        transition_desc="Update pos and time; remove target if visited; congestion penalties depend on time.",
        goal_test_desc="remaining empty within time limit (if set).",
        cost_desc="Move cost 1 + congestion_penalty(time,pos) + late_penalty(if deadlines).",
        hard_constraints=[
            "Cannot move into blocked cells",
            "Optional: must finish within max_time"
        ],
        soft_constraints=[
            "Avoid congested cells (penalty)",
            "Prefer shorter routes (penalty)"
        ],
        examples={
            "initial_state": {"pos": (0,0), "remaining": [(2,2),(4,1)], "time": 0},
            "sample_action": {"move": "up"}
        }
    )

    return {
        "Turnos (compacto)": turnos_compacto,
        "Turnos (enriquecido)": turnos_enriquecido,
        "Picking almacén (compacto)": picking_compacto,
        "Picking almacén (enriquecido)": picking_enriquecido,
    }
