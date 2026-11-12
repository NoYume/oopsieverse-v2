"""
Local page for the human feedback loop on one object: generate its damage params with the
normal pipeline, play a demo back with them, say what looked wrong, regenerate.

    python -m damagesim.llm_params.feedback_ui                  # BEHAVIOR-1K, in oopsieverse_b1k
    python -m damagesim.llm_params.feedback_ui --sim robocasa   # RoboCasa, in oopsieverse_robocasa

Needs the chosen model's API key in the environment (see ``models.py``).

Every round is a normal run directory under runs/feedback/<session>/, next to rounds.json.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

from damagesim.llm_params import generate, ground_truth as gt, prompts
from damagesim.llm_params.models import MODELS
from damagesim.llm_params.schema import PARAM_NAMES

ROOT = gt.REPO_ROOT
DEMO_DIRS = (ROOT / "demos/robocasa/teleop_data", ROOT / "oopsiebench/demos/robocasa/teleop")
PLAYBACK = ROOT / "scripts/playback_robocasa.py"
B1K_TASKS = ROOT / "oopsiebench/envs/behavior1k"
REPLAYABLE = ROOT / "data/replayable_demos.csv"
DEFAULT_MODEL = "gpt-6.1-sol"


def task_of(stem: str) -> str:
    """pick_egg_unsafe, pick_egg_2 -> pick_egg."""
    return re.sub(r"(_safe|_unsafe)?(_\d+)?$", "", stem)


def _tracked_objects() -> dict:
    tree = ast.parse((ROOT / gt.TABLES["robocasa"][0]).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "DAMAGEABLE_OBJECTS":
            return ast.literal_eval(node.value)
    raise ValueError("no DAMAGEABLE_OBJECTS table")


def objects_for_task(task: str, tracked: dict, rows: list[dict]) -> list[str]:
    """Generatable table keys the task tracks; lockbox objects stay hidden from the model."""
    allowed = {r["entry_key"] for r in rows if r["is_target"] and not r["lockbox"]}
    entry = tracked.get(task, {"names": [], "categories": []})
    names = [re.sub(r"_\d+$", "", n) for n in entry["names"] + entry["categories"]]
    return list(dict.fromkeys(n for n in names if n in allowed))


def list_robocasa_demos() -> list[dict]:
    import h5py

    tracked, rows = _tracked_objects(), gt.export("robocasa")
    out = []
    for folder in DEMO_DIRS:
        for path in sorted(folder.glob("*.hdf5")):
            task = task_of(path.stem)
            objects = [{"key": o, "health_names": []} for o in objects_for_task(task, tracked, rows)]
            if not objects:
                continue
            with h5py.File(path, "r") as f:
                ids = sorted(int(d.split("_")[1]) for d in f.get("data", {}))
            for i in ids:
                out.append({"path": str(path), "task": task, "demo_id": i, "objects": objects,
                            "label": f"{path.stem} / demo_{i}"})
    return out


def robocasa_cmd(session: dict, round_dir: Path, overrides: Path, summary: Path) -> list[str]:
    demo = session["demo"]
    return [sys.executable, "-u", str(PLAYBACK), "--input", demo["path"],
            "--output", str(round_dir / "playback_data/playback.hdf5"),
            "--env", demo["task"], "--playback", "--headless", "--camera", "task",
            "--demo_ids", str(demo["demo_id"]), "--visualize", "--metrics",
            "--param-overrides", str(overrides), "--summary-json", str(summary)]


def _strings(node: ast.AST) -> list[str]:
    return [e.value for e in getattr(node, "elts", []) if isinstance(e, ast.Constant) and isinstance(e.value, str)]


def b1k_objects(source: str, rows: list[dict]) -> list[dict]:
    """Generatable categories a B1K task tracks, with their object names in the health data.

    Reads the task file's TASK_OBJECTS and target_objects_health= without importing it (no OmniGibson
    here); only string constants are read, since some positions are expressions.
    """
    tree = ast.parse(source)
    tracked = {s for kw in ast.walk(tree) if isinstance(kw, ast.keyword) and kw.arg == "target_objects_health"
               for s in _strings(kw.value)}
    # Only the mechanical numbers are tuned, so objects that also take water or heat damage are left off.
    allowed = {r["entry_key"] for r in rows
               if r["is_target"] and not r["lockbox"] and r["damage_evaluators"] == "mechanical"}
    grouped: dict[str, list[str]] = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "TASK_OBJECTS"
                and isinstance(node.value, ast.Dict)):
            continue
        for key, entry in zip(node.value.keys, node.value.values):
            if not (isinstance(key, ast.Constant) and isinstance(entry, ast.Dict)):
                continue
            fields = {k.value: v.value for k, v in zip(entry.keys, entry.values)
                      if isinstance(k, ast.Constant) and isinstance(v, ast.Constant)}
            name, category = fields.get("name", key.value), fields.get("category")
            if name in tracked and category in allowed:
                grouped.setdefault(category, []).append(name)
    return [{"key": c, "health_names": names} for c, names in grouped.items()]


def list_b1k_demos(csv_path: Path = REPLAYABLE) -> list[dict]:
    """The omnigibson rows of data/replayable_demos.csv whose file is on disk; no HDF5 is opened."""
    rows, objects, out = gt.export("omnigibson"), {}, []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            path = ROOT / row["hdf5_path"]
            if row["backend"] != "omnigibson" or not path.exists():
                continue
            task = row["task"]
            if task not in objects:
                objects[task] = b1k_objects((B1K_TASKS / f"{task}.py").read_text(), rows)
            if not objects[task]:
                continue
            i = int(row["demo_id"].split("_")[1])
            out.append({"path": str(path), "task": task, "demo_id": i, "objects": objects[task],
                        "label": f"{task}_{row['safety']} / demo_{i}"})
    return out


def b1k_cmd(session: dict, round_dir: Path, overrides: Path, summary: Path) -> list[str]:
    demo = session["demo"]
    return [sys.executable, "-u", str(ROOT / "scripts/playback_b1k.py"), "--task_name", demo["task"],
            "--source_hdf5_path", demo["path"], "--playback_hdf5_path", str(round_dir / "playback_data/playback.hdf5"),
            "--video_dir", str(round_dir / "playback_videos"), "--demo_ids", str(demo["demo_id"]),
            "--low_resolution", "--playback", "--visualize", "--compute_metrics",
            "--param-overrides", str(overrides), "--summary-json", str(summary)]


@dataclass(frozen=True)
class Sim:
    """What differs between simulators; the server, page and feedback loop are shared."""

    backend: str                                                 # generate.Config.backend
    download: str                                                # download_demos.py --sim value
    list_demos: Callable[[], list[dict]]
    playback_cmd: Callable[[dict, Path, Path, Path], list[str]]  # (session, round_dir, overrides, summary)
    video_path: Callable[[Path, int], Path]                      # (round_dir, demo_id)


SIMS: dict[str, Sim] = {
    "omnigibson": Sim("omnigibson", "behavior1k", list_b1k_demos, b1k_cmd,
                      lambda d, i: d / "playback_videos" / f"demo_{i}_health_overlay_video.mp4"),
    "robocasa": Sim("robocasa", "robocasa", list_robocasa_demos, robocasa_cmd,
                    lambda d, i: d / f"playback_videos/playback/demo_{i}_health_overlay_video.mp4"),
}
DEFAULT_SIM = "omnigibson"  # BEHAVIOR-1K is the main supported sim


def summarize_run(run_dir: Path) -> tuple[dict | None, dict]:
    """(median params over valid samples or None, details) for a one-target run directory."""
    gens = [json.loads(l) for l in (run_dir / "generations.jsonl").read_text().splitlines()]
    ctx = [json.loads(l) for l in (run_dir / "context.jsonl").read_text().splitlines()]
    valid = [g for g in gens if g["damage_threshold"] is not None]
    errors = [r["error"] for r in ctx + gens if r["error"]]
    details = {"samples": len(valid), "exemplars": ctx[0]["exemplars"] if ctx else [],
               "reasoning": valid[0]["reasoning"] if valid else None, "error": errors[0] if errors else None}
    if not valid:
        return None, details
    return {p: float(statistics.median(g[p] for g in valid)) for p in PARAM_NAMES}, details


def generate_round(session: dict, model: str, n: int, k: int, client=None) -> dict:
    """Rerun the pipeline for the session's object, with all earlier rounds as context."""
    rounds = session["rounds"]
    feedback = prompts.feedback_block(session["object"], rounds, session["health_names"] or None) if rounds else None
    cfg = generate.Config(model=model, backend=SIMS[session["sim"]].backend, n=n, k=k, targets=(session["object"],),
                          workers=1,
                          run_id=f"round{len(rounds) + 1}_{int(time.time())}", runs_dir=Path(session["dir"]),
                          feedback=feedback)
    run_dir = generate.run(cfg, client=client)
    params, details = summarize_run(run_dir)
    if params is None:
        raise RuntimeError(f"no valid generation: {details['error']}")
    new = {"model": model, "params": params, **details, "run_dir": str(run_dir), "summary": None, "video": None, "feedback": None}
    rounds.append(new)
    return new


def playback_round(session: dict, on_line=lambda line: None) -> None:
    """Replay the session's demo with the latest round's params; fills in its summary and video."""
    sim, r = SIMS[session["sim"]], session["rounds"][-1]
    run_dir = Path(r["run_dir"])
    overrides, summary = run_dir / "params.json", run_dir / "playback_summary.json"
    overrides.write_text(json.dumps({session["object"]: r["params"]}, indent=2))
    output = run_dir / "playback_data/playback.hdf5"
    output.parent.mkdir(parents=True, exist_ok=True)  # OmniGibson's writer won't create it
    cmd = sim.playback_cmd(session, run_dir, overrides, summary)
    tail = []
    with subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        for line in proc.stdout:
            tail = (tail + [line.rstrip()])[-15:]
            on_line(line.strip())
    output.unlink(missing_ok=True)  # one camera is still hundreds of MB; the video and summary are what we keep
    video = sim.video_path(run_dir, session["demo"]["demo_id"])
    if proc.returncode != 0 or not summary.exists() or not video.exists():
        raise RuntimeError("playback failed:\n" + "\n".join(tail))
    r["summary"] = json.loads(summary.read_text())
    r["video"] = str(video)


class App:
    """One person's page: one session at a time, one job at a time."""

    def __init__(self, default_model: str, n: int, k: int, demos: list[dict], sim: str = DEFAULT_SIM):
        self.default_model, self.n, self.k, self.demos, self.sim = default_model, n, k, demos, sim
        self.session: dict | None = None
        self.busy, self.status, self.error = None, "", None
        self.version = 0
        self.lock = threading.Lock()

    def options(self) -> dict:
        return {"demos": self.demos, "default_model": self.default_model, "download": SIMS[self.sim].download,
                "models": [{"name": s.name, "has_key": bool(os.environ.get(s.key_env)), "key_env": s.key_env}
                           for s in MODELS.values()]}

    def _model_error(self, model) -> str | None:
        if model not in MODELS:
            return f"unknown model {model!r}"
        if not os.environ.get(MODELS[model].key_env):
            return f"set {MODELS[model].key_env} and restart to use {model}"
        return None

    def _save(self) -> None:
        self.version += 1
        if self.session:
            Path(self.session["dir"], "rounds.json").write_text(json.dumps(self.session, indent=2))

    def _start(self, label: str, job) -> tuple[int, dict]:
        with self.lock:
            if self.busy:
                return 409, {"error": f"still busy: {self.busy}"}
            self.busy, self.status, self.error = label, "", None

        def work():
            try:
                job()
            except (Exception, SystemExit) as exc:  # generate.run exits on config problems
                self.error = str(exc)
            self._save()
            self.busy = None

        threading.Thread(target=work, daemon=True).start()
        return 202, {}

    def generate(self, body: dict) -> tuple[int, dict]:
        demo = next((d for d in self.demos if d["path"] == body.get("path") and d["demo_id"] == body.get("demo_id")), None)
        obj = next((o for o in demo["objects"] if o["key"] == body.get("object")), None) if demo else None
        if obj is None:
            return 400, {"error": "unknown demo or object"}
        model = body.get("model")
        if error := self._model_error(model):
            return 400, {"error": error}
        session = {"sim": self.sim, "object": obj["key"], "health_names": obj["health_names"], "demo": demo, "rounds": [],
                   "dir": str(ROOT / "runs/feedback" / f"{time.strftime('%Y%m%d_%H%M%S')}_{demo['task']}_{body['object']}")}
        Path(session["dir"]).mkdir(parents=True)
        self.session = session
        return self._start("generating", lambda: generate_round(session, model, self.n, self.k))

    def playback(self) -> tuple[int, dict]:
        if not self.session or not self.session["rounds"]:
            return 400, {"error": "generate first"}
        return self._start("playing back", lambda: playback_round(self.session, self._set_status))

    def feedback(self, body: dict) -> tuple[int, dict]:
        text = str(body.get("text", "")).strip()
        if not text:
            return 400, {"error": "say what looked wrong"}
        if not self.session or not self.session["rounds"] or not self.session["rounds"][-1]["summary"]:
            return 400, {"error": "play back first"}
        model = body.get("model")
        if error := self._model_error(model):
            return 400, {"error": error}

        def job():
            self.session["rounds"][-1]["feedback"] = text
            generate_round(self.session, model, self.n, self.k)

        return self._start("regenerating with feedback", job)

    def _set_status(self, line: str) -> None:
        self.status = line

    def state(self) -> dict:
        return {"busy": self.busy, "status": self.status, "error": self.error, "version": self.version,
                "session": self.session}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, code: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _video(self, path: str) -> None:
            size = os.path.getsize(path)
            start, end, code = 0, size - 1, 200
            m = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
            if m:  # Safari will not play a video without range support
                start, end, code = int(m[1]), min(int(m[2] or size - 1), size - 1), 206
            self.send_response(code)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if m:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                self.wfile.write(f.read(end - start + 1))

        def do_GET(self):
            if self.path == "/":
                data = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/api/state":
                self._json(200, app.state())
            elif self.path == "/api/options":
                self._json(200, app.options())
            elif m := re.fullmatch(r"/video/(\d+)(\?.*)?", self.path):
                rounds = app.session["rounds"] if app.session else []
                i = int(m[1])
                if i < len(rounds) and rounds[i]["video"]:
                    self._video(rounds[i]["video"])  # only paths the server itself recorded
                else:
                    self._json(404, {"error": "no video"})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            # A web page in another tab can't send this content type here without a CORS preflight,
            # and the host check stops DNS rebinding, so only this page can spend the API quota.
            if self.headers.get("Content-Type") != "application/json" \
                    or self.headers.get("Host", "").split(":")[0] not in ("127.0.0.1", "localhost"):
                return self._json(403, {"error": "forbidden"})
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            routes = {"/api/generate": lambda: app.generate(body), "/api/playback": app.playback,
                      "/api/feedback": lambda: app.feedback(body)}
            if self.path not in routes:
                return self._json(404, {"error": "not found"})
            self._json(*routes[self.path]())

    return Handler


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Damage Param Feedback</title>
<style>
:root { color-scheme: light dark; --bg:#fff; --fg:#1c1c1e; --muted:#666; --line:#d8d8dc; --accent:#0a66c2; --err:#b3261e; --up:#c62828; --down:#1565c0; }
@media (prefers-color-scheme: dark) { :root { --bg:#17181a; --fg:#ececee; --muted:#a0a0a6; --line:#3a3b3e; --accent:#6db1f2; --err:#f2877f; --up:#ff8a80; --down:#82b1ff; } }
body { background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,sans-serif; margin:0 auto; max-width:1240px; padding:24px 16px 64px; }
h1 { font-size:20px; margin:0 0 4px; } h2 { font-size:16px; margin:0 0 8px; }
.muted { color:var(--muted); } .row { display:flex; flex-wrap:wrap; gap:10px; align-items:end; margin:16px 0; }
label { display:flex; flex-direction:column; gap:4px; font-size:13px; color:var(--muted); }
select, textarea, button { font:inherit; color:inherit; background:transparent; border:1px solid var(--line); border-radius:6px; padding:7px 10px; }
textarea { width:100%; box-sizing:border-box; min-height:64px; }
button { cursor:pointer; } button.primary { background:var(--accent); border-color:var(--accent); color:var(--bg); }
button:disabled { opacity:.45; cursor:default; }
#status { min-height:22px; overflow-wrap:anywhere; } #error { color:var(--err); white-space:pre-wrap; }
.layout { display:grid; grid-template-columns:minmax(0,1fr); grid-template-rows:auto 1fr; column-gap:32px; max-width:860px; }
.layout.has-diffs { grid-template-columns:minmax(0,1fr) 340px; max-width:none; }
#controls, #rounds { grid-column:1; min-width:0; }
#diffs { grid-column:2; grid-row:1 / span 2; align-self:start; position:sticky; top:16px; max-height:calc(100vh - 32px); overflow:auto; border-left:1px solid var(--line); padding-left:24px; }
.diff { border-top:1px solid var(--line); padding:14px 0; }
.param { margin-top:10px; font-variant-numeric:tabular-nums; }
.param .name { font-size:13px; color:var(--muted); }
.up { color:var(--up); } .down { color:var(--down); }
.note { font-size:13px; color:var(--muted); margin-bottom:12px; }
@media (max-width:900px) {
  .layout, .layout.has-diffs { grid-template-columns:minmax(0,1fr); grid-template-rows:none; }
  #diffs { grid-column:1; grid-row:auto; position:static; max-height:none; border-left:0; padding-left:0; margin-bottom:16px; }
}
.round { border-top:1px solid var(--line); padding:16px 0; }
table { border-collapse:collapse; margin:6px 0 10px; font-variant-numeric:tabular-nums; }
td, th { padding:2px 16px 2px 0; text-align:left; font-weight:400; } th { color:var(--muted); }
video { width:100%; border-radius:6px; background:#000; margin-top:6px; }
.said { border-left:3px solid var(--accent); padding-left:10px; margin:8px 0; }
</style></head><body>
<div class="layout" id="layout">
<div id="controls">
<h1>Damage param feedback</h1>
<div class="row">
  <label>Demo <select id="demo"></select></label>
  <label>Object <select id="object"></select></label>
  <label>Model <select id="model"></select></label>
  <button id="generate" class="primary">Generate properties</button>
  <button id="playback">Play back</button>
</div>
<div id="status" class="muted" role="status"></div>
<div id="error"></div>
<div id="feedbackBox" hidden>
  <h2>What looked wrong?</h2>
  <label for="feedback" class="muted" style="display:block">Regenerates with your words, the earlier numbers and what they did in simulation.</label>
  <textarea id="feedback" placeholder="that cup isn't breaking as easily as it should"></textarea>
  <div class="row"><button id="regenerate" class="primary">Regenerate with feedback</button></div>
</div>
</div>
<aside id="diffs" aria-label="What changed between rounds" hidden>
  <h2>What changed</h2>
  <div id="diffList"></div>
</aside>
<div id="rounds"></div>
</div>
<script>
const $ = id => document.getElementById(id);
let demos = [], shownVersion = -1;
const el = (tag, text, cls) => { const e = document.createElement(tag); if (text != null) e.textContent = text; if (cls) e.className = cls; return e; };
const post = async (path, body) => {
  const r = await fetch(path, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body || {})});
  if (!r.ok) $("error").textContent = (await r.json()).error;
  poll();
};
function table(head, rows) {
  const t = el("table"), h = el("tr");
  head.forEach(c => h.append(el("th", c))); t.append(h);
  rows.forEach(r => { const tr = el("tr"); r.forEach(c => tr.append(el("td", c))); t.append(tr); });
  return t;
}
const force = (p, key) => p && key in p ? p[key].toFixed(1) : "not recorded";
function renderRound(r, i, rounds) {
  const d = el("div", null, "round");
  d.append(el("h2", "Round " + (i + 1)));
  if (i > 0 && rounds[i - 1].feedback) d.append(el("div", "You said: " + rounds[i - 1].feedback, "said"));
  d.append(table(["Parameter", "Value"], Object.entries(r.params).map(([k, v]) => [k, String(+v.toPrecision(4))])));
  d.append(el("div", r.model + ": median of " + r.samples + (r.samples === 1 ? " sample" : " samples") + ". Similar objects: " + r.exemplars.join(", "), "muted"));
  if (r.reasoning) d.append(el("div", r.reasoning, "muted"));
  if (r.summary) {
    const peaks = r.summary.peak_forces;
    d.append(table(["Object", "Final health", "Peak impact", "Peak pushing"], Object.entries(r.summary.final_health).map(([o, h]) => [
      o, h.toFixed(1) + "%", force(peaks[o], "impact_forces"), force(peaks[o], "filtered_qs_forces")])));
  }
  if (r.video) {
    const v = el("video"); v.controls = true; v.preload = "metadata"; v.src = "/video/" + i + "?v=" + encodeURIComponent(r.run_dir);
    d.append(v);
  }
  if (r.feedback && i === rounds.length - 1) d.append(el("div", "Not applied yet: " + r.feedback, "said"));
  return d;
}
const fmt = v => String(+v.toPrecision(4));
function renderParam(name, before, after) {
  const row = el("div", null, "param");
  row.append(el("div", name, "name"));
  if (before === after) { row.append(el("div", fmt(after) + " (no change)", "muted")); return row; }
  const up = after > before, delta = el("div", null, up ? "up" : "down");
  const pct = before ? " (" + (up ? "+" : "") + Math.round((after - before) / before * 100) + "%)" : "";
  delta.textContent = (up ? "\u25B2 " : "\u25BC ") + fmt(before) + " \u2192 " + fmt(after) + pct;
  row.append(delta);
  return row;
}
function renderDiff(prev, cur, i) {
  const d = el("div", null, "diff");
  d.append(el("h2", "Round " + i + " \u2192 " + (i + 1)));
  if (prev.feedback) d.append(el("div", prev.feedback, "said"));
  if (prev.model !== cur.model) d.append(el("div", "Model changed: " + prev.model + " \u2192 " + cur.model, "note"));
  Object.keys(cur.params).forEach(p => d.append(renderParam(p, prev.params[p], cur.params[p])));
  return d;
}
function fillObjects() {
  const d = demos[+$("demo").value];
  $("object").replaceChildren(...(d ? d.objects : []).map(o => { const e = el("option", o.key); e.value = o.key; return e; }));
}
async function poll() {
  const s = await (await fetch("/api/state")).json();
  const rounds = s.session ? s.session.rounds : [], last = rounds[rounds.length - 1];
  $("status").textContent = s.busy ? s.busy + "... " + s.status : "";
  if (s.error) $("error").textContent = s.error; else if (s.busy) $("error").textContent = "";
  $("generate").disabled = !!s.busy || !demos.length;
  $("playback").disabled = !!s.busy || !last || !!last.summary;
  $("regenerate").disabled = !!s.busy;
  $("feedbackBox").hidden = !(last && last.summary);
  if (s.version !== shownVersion) {
    shownVersion = s.version;
    $("rounds").replaceChildren(...rounds.map(renderRound).reverse());
    const diffs = rounds.slice(1).map((r, i) => renderDiff(rounds[i], r, i + 1)).reverse();
    $("diffList").replaceChildren(...diffs);
    $("diffs").hidden = !diffs.length;
    $("layout").classList.toggle("has-diffs", diffs.length > 0);
    if (last && !last.summary) $("feedback").value = "";
  }
}
$("demo").onchange = fillObjects;
$("generate").onclick = () => { const d = demos[+$("demo").value]; post("/api/generate", {path:d.path, demo_id:d.demo_id, object:$("object").value, model:$("model").value}); };
$("playback").onclick = () => post("/api/playback");
$("regenerate").onclick = () => post("/api/feedback", {text:$("feedback").value, model:$("model").value});
(async () => {
  const options = await (await fetch("/api/options")).json();
  demos = options.demos;
  $("model").replaceChildren(...options.models.map(m => {
    const e = el("option", m.has_key ? m.name : m.name + " (no " + m.key_env + ")"); e.value = m.name; e.disabled = !m.has_key; return e;
  }));
  $("model").value = options.default_model;
  $("demo").replaceChildren(...demos.map((d, i) => { const e = el("option", d.label); e.value = i; return e; }));
  fillObjects();
  if (!demos.length) $("error").textContent = "No demos found. Run: python scripts/download_demos.py --sim " + options.download;
  poll(); setInterval(poll, 1000);
})();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--sim", default=DEFAULT_SIM, choices=list(SIMS), help="simulator the demos are played back in")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS), help="model selected when the page opens")
    ap.add_argument("--n", type=int, default=5, help="generations per round; their median is played back")
    ap.add_argument("--k", type=int, default=5, help="similar objects shown to the model")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    if not any(os.environ.get(s.key_env) for s in MODELS.values()):
        raise SystemExit("no API key found; set one in the environment, such as OPENAI_API_KEY")
    app = App(args.model, args.n, args.k, SIMS[args.sim].list_demos(), sim=args.sim)
    print(f"Open http://127.0.0.1:{args.port}  ({len(app.demos)} demos)")
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app)).serve_forever()


if __name__ == "__main__":
    main()
