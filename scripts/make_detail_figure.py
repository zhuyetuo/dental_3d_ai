"""docs/images/generation_detail.png：学习模型的平滑牙冠 vs 换上真牙细节后 vs 真牙。病例取各类里学习模型误差中位数的。"""
import csv, json, sys, tempfile
from pathlib import Path
import numpy as np, trimesh

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() and Path(__file__).parent.name == "scripts" else Path.home() / "zyt/dental_3d_ai"
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "ml"))
import make_result_figures as F
from dental_ai.eval_generation import _Fixed
from dental_ai.generation.learned import LearnedGenerator
from dental_ai.generation.mirror_template import MirrorTemplateGenerator
from dental_ai.pipeline import run

rows_csv = [r for r in csv.DictReader(open(ROOT / "runs/gen_compare/learned/generation.csv")) if r.get("chamfer_l1")]
det = {(r["scan"], r["scenario"], r["fdi"]): r for r in csv.DictReader(open(ROOT / "runs/gen_compare/detail3/generation.csv")) if r.get("chamfer_l1")}


def pick(q, cond):
    sub = sorted((r for r in rows_csv if r["scenario"] == "single" and cond(r) and (r["scan"], r["scenario"], r["fdi"]) in det), key=lambda r: float(r["chamfer_l1"]))
    return sub[int(q * (len(sub) - 1))]


src = lambda r: det[(r["scan"], r["scenario"], r["fdi"])]["source"]
picks = [("upper first molar\n(detail from FDI16)", pick(0.5, lambda r: int(r["fdi"]) in (16, 26) and "fdi16" in src(r))),
         ("lower first molar", pick(0.5, lambda r: int(r["fdi"]) in (36, 46) and "detail" in src(r))),
         ("premolar", pick(0.5, lambda r: int(r["fdi"]) % 10 in (4, 5) and "detail" in src(r)))]
model = str(ROOT / "models/generation/toothgen_open.pt")
gens = {"plain": LearnedGenerator(model, fallback=MirrorTemplateGenerator()),
        "detail": LearnedGenerator(model, fallback=MirrorTemplateGenerator(), detail_dir=str(ROOT / "data/gen_library"))}
gens["plain"].detail_dir = None
rows, titles, notes = [], [], []
for title, r in picks:
    fdi = int(r["fdi"])
    mesh0, gt0, jaw = F.scan(r["scan"])
    teeth = tuple(int(t) for t in r["teeth"].split())
    mesh, labels = F.simulate_missing(mesh0, gt0, jaw, teeth)
    up = F.estimate_arch_frame(mesh0, gt0, jaw).occlusal
    real = F.tooth_mesh(mesh0, gt0, fdi)
    center = real.vertices.mean(axis=0)
    cam = F.camera(up, center - mesh0.vertices[gt0 > 0].mean(axis=0), 55)
    col = lambda lab: np.tile(F.GUM, (len(lab), 1)) * (lab[:, None] == 0) + np.tile(F.IVORY, (len(lab), 1)) * (lab[:, None] > 0)
    base = F.sample(mesh, col(labels), per_mm2=700)
    ims = []
    for name in ("plain", "detail"):
        tmp = Path(tempfile.mkdtemp())
        rep = run(mesh, jaw, _Fixed(labels), tmp, generator=gens[name])
        g = next(x for x in rep["generated"] if x["fdi"] == fdi)
        solid = trimesh.load(tmp / g["file"])
        ims.append(F.render([base, F.sample(solid, np.tile(F.BLUE, (len(solid.vertices), 1)), per_mm2=700)], cam, center, 9))
    ims.append(F.render([F.sample(mesh0, col(gt0), per_mm2=700)], cam, center, 9))
    rows.append(ims)
    d = det[(r["scan"], r["scenario"], r["fdi"])]
    titles.append(f"{title}\ntooth {fdi}")
    notes.append({"scan": r["scan"], "fdi": fdi, "plain_chamfer_mm": round(float(r["chamfer_l1"]), 3), "detail_chamfer_mm": round(float(d["chamfer_l1"]), 3), "source": d["source"]})
F.grid(rows, ["learned model (smooth)", "+ real-crown detail", "real tooth"], titles, ROOT / "docs/images/generation_detail.png", panel=3.0)
print(json.dumps(notes, indent=1))
