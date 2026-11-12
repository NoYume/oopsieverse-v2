"""
Helpers the playback scripts share with the feedback page: apply generated mechanical params
to a damage table, and boil a playback down to final health and peak forces per object.

No simulator imports, so both are testable without RoboCasa or OmniGibson.
"""

from __future__ import annotations

from damagesim.llm_params.schema import PARAM_NAMES


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
