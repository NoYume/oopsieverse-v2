"""The feedback loop without a simulator or network: the next round's prompts carry the last round."""

import json
from types import SimpleNamespace

from damagesim.llm_params import feedback_ui as ui
from damagesim.llm_params import ground_truth as gt


class FakeClient:
    def __init__(self):
        self.prompts = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.prompts.append(kw["messages"][-1]["content"])
        js = kw["response_format"]["json_schema"]
        if js["name"] == "Selection":
            reply = {"similar": js["schema"]["properties"]["similar"]["items"]["enum"][:2]}
        else:
            reply = {"damage_threshold": 5.0 + len(self.prompts), "damage_scale": 1.0, "impact_damage_sensitivity": 1.0,
                     "qs_damage_sensitivity": 0.5, "reasoning": "fragile"}
        msg = SimpleNamespace(content=json.dumps(reply))
        return SimpleNamespace(id="r", choices=[SimpleNamespace(message=msg)], usage=None)


def test_task_of():
    assert [ui.task_of(s) for s in ("pick_egg", "pick_egg_2", "pick_egg_unsafe", "prepare_coffee_safe")] == [
        "pick_egg", "pick_egg", "pick_egg", "prepare_coffee"]


def test_objects_strip_instances_and_hide_lockbox_and_robot():
    rows = gt.export("robocasa")
    tracked = {"t": {"categories": ["agent", "microwave"], "names": ["wine_1", "wine_2", "milk", "distr_object"]}}
    assert ui.objects_for_task("t", tracked, rows) == ["wine", "microwave"]
    assert ui.objects_for_task("missing", tracked, rows) == []


def test_feedback_round_sends_earlier_numbers_result_and_words(tmp_path):
    client = FakeClient()
    session = {"object": "cup", "dir": str(tmp_path), "rounds": []}

    first = ui.generate_round(session, "gemini-3.5-flash-lite", n=3, k=2, client=client)
    assert first["samples"] == 3 and first["params"]["damage_threshold"] == 8.0  # median of 7, 8, 9
    assert not any("Earlier attempts" in p for p in client.prompts)

    first["summary"] = {"final_health": {"cup": 100.0}, "peak_forces": {"cup": {"impact_forces": 40.0}}}
    first["feedback"] = "that cup isn't breaking as easily as it should"
    client.prompts.clear()
    ui.generate_round(session, "gemini-3.5-flash-lite", n=1, k=2, client=client)

    assert len(session["rounds"]) == 2 and len(client.prompts) == 2  # the whole pipeline: select, then generate
    for prompt in client.prompts:
        assert "isn't breaking" in prompt and "damage_threshold=8.0" in prompt and "peak impact force 40.0" in prompt


def test_requested_model_must_be_known_and_have_a_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
    demo = {"path": "d.hdf5", "demo_id": 0, "task": "pick_egg", "objects": ["egg"]}
    app = ui.App("gemini-3.5-flash-lite", 1, 1, [demo])
    ask = {"path": "d.hdf5", "demo_id": 0, "object": "egg"}

    code, body = app.generate({**ask, "model": "claude-haiku-5-5"})
    assert code == 400 and "CLAUDE_API_KEY" in body["error"]
    assert app.generate({**ask, "model": "nope"})[0] == 400
    assert app.session is None
    available = {m["name"]: m["has_key"] for m in app.options()["models"]}
    assert available["gemini-3.5-flash-lite"] and not available["claude-haiku-5-5"]
