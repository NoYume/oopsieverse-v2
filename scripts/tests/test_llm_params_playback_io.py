"""Sim-free helpers shared by the playback scripts: param overrides and the playback summary."""

import pytest

from damagesim.llm_params import playback_io as io


def _params():
    return {"bag_of_flour": {"mechanical": {"damage_threshold": 150.0, "damage_scale": 100.0}},
            "agent": {"thermal": {"heating_threshold": 1.0}}}


def test_apply_overrides_updates_mechanical_only():
    p = _params()
    io.apply_overrides(p, {"bag_of_flour": {"damage_threshold": 20.0}})
    assert p["bag_of_flour"]["mechanical"] == {"damage_threshold": 20.0, "damage_scale": 100.0}
    assert p["agent"] == {"thermal": {"heating_threshold": 1.0}}


@pytest.mark.parametrize("bad, word", [({"bag_of_flor": {"damage_threshold": 1.0}}, "bag_of_flor"),
                                       ({"agent": {"damage_threshold": 1.0}}, "agent"),
                                       ({"bag_of_flour": {"heating_threshold": 1.0}}, "heating_threshold")])
def test_apply_overrides_rejects_typos_before_changing_anything(bad, word):
    p = _params()
    with pytest.raises(ValueError, match=word):
        io.apply_overrides(p, {"bag_of_flour": {"damage_threshold": 5.0}, **bad})
    assert p == _params()


def test_summarize_peaks_over_time_links_and_demos():
    forces = [{"book@base_link": {"impact_forces": [1.0, 5.0], "filtered_qs_forces": [0.0, 3.0]},
               "robot0@eef_link": {"impact_forces": [9.0], "filtered_qs_forces": []}},
              {"book@base_link": {"impact_forces": [7.0], "filtered_qs_forces": [2.0]}}]
    out = io.summarize({"book": [80.0, 60.0], "robot0": [100.0]}, forces)
    assert out["final_health"] == {"book": 70.0, "robot0": 100.0}
    assert out["peak_forces"]["book"] == {"impact_forces": 7.0, "filtered_qs_forces": 3.0}
    assert out["peak_forces"]["robot0"] == {"impact_forces": 9.0, "filtered_qs_forces": 0.0}
