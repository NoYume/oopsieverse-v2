"""
The two prompts: pick similar objects, then generate the numbers. Prompt changes are
one-file diffs here; keep them short.
"""

from __future__ import annotations

from damagesim.llm_params.schema import PARAM_NAMES

SIMULATOR_NOTE = {
    "robocasa": "These numbers are for the RoboCasa (MuJoCo) simulator, in its own internal units. "
                "Stay on the scale of the examples.",
    "omnigibson": "These numbers are for the BEHAVIOR-1K (OmniGibson) simulator, in its own internal units. "
                  "Stay on the scale of the examples.",
}

# Per step: D = impact_damage_sensitivity * impact force + qs_damage_sensitivity * pushing force;
# health lost = max(0, D - damage_threshold) * damage_scale (damagesim/core/evaluators/mechanical.py).
PARAM_MEANINGS = {
    "damage_threshold": "How hard a hit or push has to be before any damage happens.",
    "damage_scale": "How fast health drops once the threshold is passed.",
    "impact_damage_sensitivity": "How much sudden impacts (drops, collisions) count.",
    "qs_damage_sensitivity": "How much steady pushing or squeezing counts.",
}


def feedback_block(target_key: str, rounds: list[dict], health_names: list[str] | None = None) -> str:
    """Earlier attempts at one object: their numbers, what they did in simulation, what the human said.

    ``health_names`` are the target's names in the playback summary when they differ from its table key
    (B1K's bag_of_flour is "book"); without them the target is found by substring.
    """
    lines = ["Earlier attempts at this object, each played back in simulation and judged by a human. "
             "Per step: D = impact_damage_sensitivity * impact force + qs_damage_sensitivity * pushing force; "
             "health lost = max(0, D - damage_threshold) * damage_scale. Health starts at 100.", ""]
    for i, r in enumerate(rounds, start=1):
        lines.append(f"Attempt {i}: " + ", ".join(f"{p}={r['params'][p]!r}" for p in PARAM_NAMES))
        health = (r.get("summary") or {}).get("final_health", {})
        peaks = (r.get("summary") or {}).get("peak_forces", {})
        for obj in health_names or [o for o in health if target_key in o.lower()] or list(health):
            if obj not in health:
                lines.append(f"Result for {obj}: not recorded")
                continue
            f = peaks.get(obj, {})
            force = lambda key: f"{f[key]:.1f}" if key in f else "not recorded"  # a missing force is not a zero
            lines.append(f"Result for {obj}: final health {health[obj]:.1f}, peak impact force "
                         f"{force('impact_forces')}, peak pushing force {force('filtered_qs_forces')}")
        if r.get("feedback"):
            lines.append(f"Human feedback: {r['feedback']}")
        lines.append("")
    lines.append("Change the numbers so the object behaves as the latest feedback asks; "
                 "where the feedback and the similar objects disagree, follow the feedback.")
    return "\n".join(lines)


def selection_messages(backend: str, target_name: str, pool_names: list[str], k: int,
                       feedback: str | None = None) -> list[dict]:
    listing = "\n".join(f"- {n}" for n in pool_names)
    note = f"\n\n{feedback}" if feedback else ""
    return [
        {"role": "system", "content": "You help set damage parameters for objects in a robot simulator."},
        {"role": "user", "content": (
            f"Object: {target_name}\n\n"
            f"From the list below, pick the {k} objects that would break most like a {target_name} "
            f"(material, fragility, size). Use names from the list only.\n\n{listing}{note}"
        )},
    ]


def _exemplar_line(row: dict) -> str:
    return f"{row['display_name']}: " + ", ".join(f"{p}={float(row[p])!r}" for p in PARAM_NAMES)


def generation_messages(backend: str, target_name: str, exemplars: list[dict],
                        feedback: str | None = None) -> list[dict]:
    meanings = "\n".join(f"- {p}: {PARAM_MEANINGS[p]}" for p in PARAM_NAMES)
    examples = "\n".join(_exemplar_line(r) for r in exemplars)
    note = f"{feedback}\n\n" if feedback else ""
    return [
        {"role": "system", "content": "You set damage parameters for objects in a robot simulator. "
                                      "Answer with the JSON schema only; never ask a question."},
        {"role": "user", "content": (
            f"{SIMULATOR_NOTE[backend]}\n\nParameters:\n{meanings}\n\n"
            f"Hand-tuned values for similar objects:\n{examples}\n\n"
            f"{note}Give the four parameters for: {target_name}\n"
            "Also give a one-sentence reasoning."
        )},
    ]
