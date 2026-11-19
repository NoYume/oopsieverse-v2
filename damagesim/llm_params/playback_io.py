"""
Helpers the playback scripts share with the feedback page: apply generated mechanical params
to a damage table, and boil a playback down to final health and peak forces per object.

No simulator imports, so both are testable without RoboCasa or OmniGibson.
"""

from __future__ import annotations

import json
from pathlib import Path

from damagesim.llm_params.schema import PARAM_NAMES

# Mechanical params saved from the feedback page; they win over the hand-tuned RoboCasa table.
TUNED_PARAMS_PATH = Path(__file__).resolve().parents[2] / "data/tuned_params.json"


def apply_overrides(params: dict, overrides: dict[str, dict[str, float]]) -> None:
    """Replace mechanical kwargs in place; every name is checked before anything changes."""
    for category, values in overrides.items():
        if "mechanical" not in params.get(category, {}):
            raise ValueError(f"no mechanical params for category {category!r}")
        unknown = sorted(set(values) - set(PARAM_NAMES))
        if unknown:
            raise ValueError(f"{category}: unknown mechanical params {unknown}; allowed: {list(PARAM_NAMES)}")
    for category, values in overrides.items():
        params[category]["mechanical"].update(values)


def apply_tuned(params: dict, path: Path = TUNED_PARAMS_PATH) -> None:
    """Apply the saved tuned params to a damage table in place; a missing file changes nothing."""
    if path.exists():
        apply_overrides(params, json.loads(path.read_text()))


def save_tuned(key: str, values: dict[str, float], path: Path = TUNED_PARAMS_PATH) -> None:
    """Add or replace one object's tuned mechanical params, keeping everyone else's."""
    unknown = sorted(set(values) - set(PARAM_NAMES))
    if unknown:
        raise ValueError(f"{key}: unknown mechanical params {unknown}")
    tuned = json.loads(path.read_text()) if path.exists() else {}
    tuned[key] = {p: float(values[p]) for p in PARAM_NAMES if p in values}
    path.write_text(json.dumps(dict(sorted(tuned.items())), indent=2) + "\n")


def summarize(final_healths: dict[str, list[float]],
              forces: list[dict[str, dict[str, list[float]]]]) -> dict:
    """{"final_health": {obj: mean over demos}, "peak_forces": {obj: {force_key: max}}}.

    ``forces`` has one item per demo, keyed "obj@link"; the max runs over time, links and demos.
    """
    peaks: dict[str, dict[str, float]] = {}
    for demo in forces:
        for obj_link, series in demo.items():
            obj = peaks.setdefault(obj_link.split("@", 1)[0], {})
            for key, values in series.items():
                obj[key] = max([obj.get(key, 0.0), *map(float, values)])
    return {"final_health": {obj: sum(v) / len(v) for obj, v in final_healths.items() if v},
            "peak_forces": peaks}
