"""The feedback loop without a simulator or network: the next round's prompts carry the last round."""

import json
import sys
from pathlib import Path
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
    session = {"sim": "robocasa", "object": "cup", "health_names": [], "dir": str(tmp_path), "rounds": []}

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
    demo = {"path": "d.hdf5", "demo_id": 0, "task": "pick_egg", "objects": [{"key": "egg", "health_names": []}]}
    app = ui.App("gemini-3.5-flash-lite", 1, 1, [demo], sim="robocasa")
    ask = {"demos": [{"path": "d.hdf5", "demo_id": 0}], "object": "egg"}

    code, body = app.generate({**ask, "model": "claude-haiku-5-5"})
    assert code == 400 and "CLAUDE_API_KEY" in body["error"]
    assert app.generate({**ask, "model": "nope"})[0] == 400
    assert app.session is None
    available = {m["name"]: m["has_key"] for m in app.options()["models"]}
    assert available["gemini-3.5-flash-lite"] and not available["claude-haiku-5-5"]


def test_defaults_are_b1k():
    from damagesim.llm_params import generate
    assert ui.DEFAULT_SIM == "omnigibson" and generate.Config().backend == "omnigibson"
    assert ui.DEFAULT_MODEL == "gpt-6.1-sol"
    assert ui.App("gpt-6.1-sol", 1, 1, []).sim == "omnigibson"


def test_robocasa_playback_cmd_is_sopans(tmp_path):
    s = {"demo": {"path": "d.hdf5", "task": "pick_egg", "demo_id": 3}}
    cmd = ui.SIMS["robocasa"].playback_cmd(s, tmp_path, tmp_path / "params.json", tmp_path / "s.json")
    assert cmd[2:] == [str(ui.ROOT / "scripts/playback_robocasa.py"), "--input", "d.hdf5",
                       "--output", str(tmp_path / "playback_data/playback.hdf5"), "--env", "pick_egg", "--playback",
                       "--headless", "--camera", "task", "--demo_ids", "3", "--visualize", "--metrics",
                       "--param-overrides", str(tmp_path / "params.json"), "--summary-json", str(tmp_path / "s.json")]


def test_playback_round_plays_every_demo_and_labels_results(tmp_path, monkeypatch):
    rd = tmp_path / "round1"; rd.mkdir()
    script = ("import json,pathlib,sys; d=pathlib.Path(sys.argv[1]); "
              "assert (d/'playback_data').is_dir(); (d/'playback_data/playback.hdf5').write_text('x'); "
              "(d/'v.mp4').write_text('v'); "
              "pathlib.Path(sys.argv[2]).write_text(json.dumps({'final_health': {'egg': float(sys.argv[3])}, 'peak_forces': {}}))")
    fake = ui.Sim("robocasa", "robocasa", list,
                  lambda s, d, o, sm: [sys.executable, "-c", script, str(d), str(sm), str(s["demo"]["demo_id"] * 50)],
                  lambda d, i: d / "v.mp4")
    monkeypatch.setitem(ui.SIMS, "fake", fake)
    demos = [{"demo_id": 0, "label": "pick_egg_safe / demo_0"}, {"demo_id": 1, "label": "pick_egg_unsafe / demo_1"}]
    session = {"sim": "fake", "object": "egg", "demos": demos,
               "rounds": [{"params": {"damage_threshold": 1.0}, "run_dir": str(rd), "playbacks": None}]}
    lines = []
    ui.playback_round(session, lines.append)
    plays = session["rounds"][0]["playbacks"]
    assert [p["label"] for p in plays] == [d["label"] for d in demos]
    assert [p["summary"]["final_health"]["egg"] for p in plays] == [0.0, 50.0]
    assert plays[1]["video"] == str(rd / "demo1/v.mp4")
    assert not (rd / "demo0/playback_data/playback.hdf5").exists()
    assert json.loads((rd / "params.json").read_text()) == {"egg": {"damage_threshold": 1.0}}


def test_feedback_block_reports_each_demo():
    from damagesim.llm_params import prompts
    params = {"damage_threshold": 60.0, "damage_scale": 1.0, "impact_damage_sensitivity": 1.0, "qs_damage_sensitivity": 0.2}
    rounds = [{"params": params, "feedback": "should crack when dropped", "playbacks": [
        {"label": "pick_egg_safe / demo_0", "summary": {"final_health": {"egg": 100.0}, "peak_forces": {}}},
        {"label": "pick_egg_unsafe / demo_0", "summary": {"final_health": {"egg": 100.0, "PandaOmron": 90.0},
                                                          "peak_forces": {"egg": {"impact_forces": 80.0}}}}]}]
    block = prompts.feedback_block("egg", rounds)
    assert "Result for egg in demo pick_egg_safe / demo_0: final health 100.0" in block
    assert "Result for egg in demo pick_egg_unsafe / demo_0: final health 100.0, peak impact force 80.0" in block
    assert "PandaOmron" not in block


def test_tuned_json_overrides_only_saved_objects(tmp_path):
    from damagesim.llm_params.playback_io import apply_tuned, save_tuned
    path = tmp_path / "tuned.json"
    table = {"cup": {"friction": 1, "mechanical": {"damage_threshold": 60.0, "damage_scale": 2.0}},
             "egg": {"mechanical": {"damage_threshold": 25.0, "damage_scale": 2.5}}}
    apply_tuned(table, path)  # no file yet: nothing changes
    assert table["cup"]["mechanical"]["damage_threshold"] == 60.0

    save_tuned("cup", {"damage_threshold": 45.0, "damage_scale": 0.8}, path)
    save_tuned("bowl", {"damage_threshold": 30.0}, path)
    save_tuned("cup", {"damage_threshold": 40.0, "damage_scale": 0.8}, path)  # a later save replaces it
    assert json.loads(path.read_text()) == {"bowl": {"damage_threshold": 30.0},
                                            "cup": {"damage_threshold": 40.0, "damage_scale": 0.8}}

    del table  # bowl isn't in this table, so use one that has every saved object
    table = {"cup": {"friction": 1, "mechanical": {"damage_threshold": 60.0, "damage_scale": 2.0}},
             "bowl": {"mechanical": {"damage_threshold": 50.0}},
             "egg": {"mechanical": {"damage_threshold": 25.0, "damage_scale": 2.5}}}
    apply_tuned(table, path)
    assert table["cup"] == {"friction": 1, "mechanical": {"damage_threshold": 40.0, "damage_scale": 0.8}}
    assert table["egg"]["mechanical"] == {"damage_threshold": 25.0, "damage_scale": 2.5}


def test_robocasa_table_reads_the_tuned_file():
    import inspect
    from damagesim.robosuite.params import damage_params
    assert "apply_tuned(OBJECT_PARAMS)" in inspect.getsource(damage_params)


def _app_with_round(monkeypatch, playbacks, sim="robocasa"):
    app = ui.App("gemini-3.5-flash-lite", 1, 1, [], sim=sim)
    app.session = {"sim": sim, "object": "egg", "dir": ".", "rounds": [{"params": {"damage_threshold": 9.0},
                                                                         "playbacks": playbacks}]}
    saved = []
    monkeypatch.setattr(ui, "save_params", lambda key, params: saved.append((key, params)))
    monkeypatch.setattr(app, "_save", lambda: None)
    return app, saved


def test_save_needs_a_played_back_robocasa_round(monkeypatch):
    app, saved = _app_with_round(monkeypatch, None)
    assert app.save()[0] == 400 and not saved
    app, saved = _app_with_round(monkeypatch, [{"label": "x"}], sim="omnigibson")
    assert app.save()[0] == 400 and not saved
    app, saved = _app_with_round(monkeypatch, [{"label": "x"}])
    assert app.save() == (200, {}) and saved == [("egg", {"damage_threshold": 9.0})]
    assert app.session["rounds"][-1]["saved"]


def test_generate_takes_demos_of_one_task(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    obj = [{"key": "egg", "health_names": []}]
    demos = [{"path": "a.hdf5", "demo_id": 0, "task": "pick_egg", "objects": obj, "label": "a"},
             {"path": "b.hdf5", "demo_id": 0, "task": "serve_pastry", "objects": obj, "label": "b"}]
    app = ui.App("gemini-3.5-flash-lite", 1, 1, demos, sim="robocasa")
    ask = {"object": "egg", "model": "gemini-3.5-flash-lite"}
    assert app.generate({**ask, "demos": []})[0] == 400
    assert app.generate({**ask, "demos": [{"path": "a.hdf5", "demo_id": 0}, {"path": "b.hdf5", "demo_id": 0}]})[0] == 400
    assert app.generate({**ask, "demos": [{"path": "zzz.hdf5", "demo_id": 0}]})[0] == 400
    assert app.session is None


B1K_ROWS = gt.export("omnigibson")


def _b1k(task):
    return ui.b1k_objects((ui.B1K_TASKS / f"{task}.py").read_text(), B1K_ROWS)


def test_b1k_objects_shelve_item_maps_category_to_health_name():
    objs = _b1k("shelve_item")
    assert objs[0] == {"key": "bag_of_flour", "health_names": ["book"]}
    assert [o["key"] for o in objs] == ["bag_of_flour", "bottle_of_wine", "wineglass", "bottle_of_beer"]


def test_b1k_objects_skip_expression_positions_and_untracked():
    assert _b1k("fill_bowl") == [{"key": "bowl", "health_names": ["bowl"]}]
    assert _b1k("add_firewood") == []


def test_b1k_objects_groups_names_and_defaults_name_to_key():
    src = ('TASK_OBJECTS = {"a": {"category": "egg", "position": [1 + 1, 0, 0]},\n'
           '                "b": {"name": "egg_2", "category": "egg"}}\n'
           'def get_task_config():\n    return dict(target_objects_health=[ROBOT_NAME, "a", "egg_2"])\n')
    assert ui.b1k_objects(src, B1K_ROWS) == [{"key": "egg", "health_names": ["a", "egg_2"]}]


def test_list_b1k_demos_reads_csv_and_skips_missing_files(tmp_path):
    there = tmp_path / "shelve_item_safe.hdf5"; there.write_text("")
    csv_path = tmp_path / "demos.csv"
    csv_path.write_text("backend,task,demo_id,hdf5_path,safety,num_steps\n"
                        f"omnigibson,shelve_item,demo_1,{there},safe,10\n"
                        f"omnigibson,shelve_item,demo_0,{tmp_path / 'gone.hdf5'},unsafe,10\n"
                        f"robocasa,pick_egg,demo_0,{there},safe,10\n")
    [d] = ui.list_b1k_demos(csv_path)
    assert (d["task"], d["demo_id"], d["label"], d["path"]) == ("shelve_item", 1, "shelve_item_safe / demo_1", str(there))
    assert d["objects"][0]["health_names"] == ["book"]


def test_b1k_playback_cmd(tmp_path):
    s = {"demo": {"path": "x.hdf5", "task": "shelve_item", "demo_id": 0}}
    cmd = ui.SIMS["omnigibson"].playback_cmd(s, tmp_path, tmp_path / "p.json", tmp_path / "s.json")
    assert cmd[2:] == [str(ui.ROOT / "scripts/playback_b1k.py"), "--task_name", "shelve_item", "--source_hdf5_path",
                       "x.hdf5", "--playback_hdf5_path", str(tmp_path / "playback_data/playback.hdf5"), "--video_dir",
                       str(tmp_path / "playback_videos"), "--demo_ids", "0", "--low_resolution", "--playback",
                       "--visualize", "--compute_metrics", "--param-overrides", str(tmp_path / "p.json"),
                       "--summary-json", str(tmp_path / "s.json")]


def test_b1k_feedback_round_uses_omnigibson_and_health_names(tmp_path):
    client = FakeClient()
    session = {"sim": "omnigibson", "object": "bag_of_flour", "health_names": ["book"], "dir": str(tmp_path), "rounds": []}
    first = ui.generate_round(session, "gpt-6.1-sol", n=1, k=2, client=client)
    assert json.loads((Path(first["run_dir"]) / "config.json").read_text())["config"]["backend"] == "omnigibson"
    first["summary"] = {"final_health": {"book": 100.0, "robot0": 90.0}, "peak_forces": {"book": {"impact_forces": 12.0}}}
    first["feedback"] = "should tear when dropped"
    client.prompts.clear()
    ui.generate_round(session, "gpt-6.1-sol", n=1, k=2, client=client)
    assert all("Result for book" in p and "robot0" not in p for p in client.prompts)


def test_b1k_objects_skip_categories_with_non_mechanical_damage():
    assert _b1k("pour_water") == []  # laptop's water damage is electrical; the page tunes mechanical numbers only
