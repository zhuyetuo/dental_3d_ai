"""真实缺牙测试：Bite2Text 的医生报告里写明缺失的牙，分割 + 缺牙检测能不能找出来。

Bite2Text（ditto.ing.unimore.it/bite2text，CC BY-NC-SA 4.0）：约 1000 位患者，上下颌口扫 + 医生写的咬合报告。
报告不是缺牙清单，只在缺牙影响咬合评估时才提（“因缺失 46 无法评估磨牙关系”“13 不在牙弓内”），所以：
- 报告里写明缺失的恒牙 = 确认缺失（可以算检出率）；
- 报告没提的牙不能当作“在”，所以算不了误报率，只能在“报告完全没提缺牙”的对照组里看报了多少。
儿童病例里恒牙没萌出、乳牙还在原位的情况单独统计（报告同时提到对应的乳牙在位）。

    python -m dental_ai.eval_real_missing extract --zip Bite2Text.zip --out ../runs/real_missing/truth.json
    python -m dental_ai.eval_real_missing run --zip Bite2Text.zip --truth ../runs/real_missing/truth.json \\
        --model ../models/segmentation/deep.pt --out ../runs/real_missing --workers 6
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

NUM = r"\d{2}(?:\s*(?:,|-|–|and|&)\s*\d{2})*(?:\s*,?\s*and\s+\d{2})?"
TOOTH = r"(?:the\s+)?(?:teeth|tooth|elements?|dental elements?)?\s*"
PATTERNS = [
    re.compile(rf"(?:absence|lack|loss|agenesis|extraction)s?\s+(?:(?:in|from|within)\s+the\s+(?:dental\s+)?arch(?:es)?\s+)?of\s+{TOOTH}({NUM})", re.I),
    re.compile(rf"{TOOTH}({NUM})\s+(?:is|are|was|were)\s+(?:not\s+present|absent|missing|not\s+in\s+the\s+arch|lacking|agenetic)", re.I),
    re.compile(rf"missing\s+{TOOTH}({NUM})", re.I),
    re.compile(rf"(?:since|because|as)\s+{TOOTH}({NUM})\s+(?:is|are)\s+(?:not\s+present|absent|missing)", re.I),
]
PRESENT = re.compile(rf"(?:{TOOTH}({NUM})\s+(?:(?:is|are)\s+)?(?:still\s+)?(?:present|in\s+the\s+arch|retained|persist)|(?:presence|persistence|retention)\s+of\s+(?:the\s+)?(?:deciduous\s+|primary\s+)?{TOOTH}({NUM}))", re.I)
PRIMARY_OF = {q * 10 + p: (q + 4) * 10 + p for q in (1, 2, 3, 4) for p in range(1, 6)}  # 恒牙 → 对应的乳牙


def numbers(text: str) -> list[int]:
    return [int(n) for n in re.findall(r"\d{2}", text)]


def parse_report(text: str) -> tuple[set[int], set[int], list[str]]:
    """→ 写明缺失的牙、写明在位的牙、涉及的句子。"""
    missing, present, sentences = set(), set(), []
    for s in re.split(r"(?<=[.;])\s+", text):
        found = set()
        for pat in PATTERNS:
            for m in pat.finditer(s):
                found |= set(numbers(m.group(1)))
        if found:
            sentences.append(s.strip())
            missing |= found
        for m in PRESENT.finditer(s):
            if not re.search(r"not\s+(?:present|in\s+the\s+arch)", m.group(0), re.I):
                present |= set(numbers(m.group(1) or m.group(2)))
    return missing, present - missing, sentences


def extract(zip_path: Path) -> dict:
    z = zipfile.ZipFile(zip_path)
    per: dict[str, dict] = {}
    for name in z.namelist():
        if "/reports_ios_en/" not in name or not name.endswith(".txt"):
            continue
        pid = name.split("/")[0]
        miss, pres, sents = parse_report(z.read(name).decode("utf8", "replace"))
        d = per.setdefault(pid, {"reports": 0, "votes": Counter(), "present": set(), "sentences": []})
        d["reports"] += 1
        d["votes"].update(miss)
        d["present"] |= pres
        d["sentences"] += sents
    out = {}
    for pid, d in sorted(per.items()):
        perm = sorted(t for t in d["votes"] if 11 <= t <= 47 and t % 10 in range(1, 8))
        out[pid] = {
            "reports": d["reports"],
            "missing": perm,
            "votes": {str(t): d["votes"][t] for t in perm},
            # 报告说对应的乳牙还在位：恒牙没萌出，位置被乳牙占着
            "primary_in_place": sorted(t for t in perm if PRIMARY_OF.get(t) in d["present"]),
            "other_mentions": sorted(t for t in d["votes"] if t not in perm),
            "sentences": d["sentences"],
        }
    return out


def jaw_of(fdi: int) -> str:
    return "upper" if fdi // 10 in (1, 2) else "lower"


_SEG = None


def _segmenter(model: str, orient: bool):
    global _SEG
    if _SEG is None:
        from .orientation import OrientedSegmenter
        from .segmentation.deep import DeepSegmenter

        seg = DeepSegmenter(model)
        _SEG = OrientedSegmenter(seg) if orient else seg
    return _SEG


def _run_one(args):
    zip_path, pid, jaw, model, orient = args
    import trimesh

    from .pipeline import check_arch_order, correct_numbering, detect_missing, merge_split_teeth, remove_small_islands

    row = {"patient": pid, "jaw": jaw}
    try:
        z = zipfile.ZipFile(zip_path)
        with tempfile.NamedTemporaryFile(suffix=".stl") as tmp:
            tmp.write(z.read(f"{pid}/ios/ios_{jaw}.stl"))
            tmp.flush()
            mesh = trimesh.load(tmp.name)
        seg = _segmenter(model, orient)
        labels = seg.predict(mesh, jaw)
        orientation = getattr(seg, "last", None)
        numbering = correct_numbering(mesh, labels, jaw, canonical_vertices=orientation.apply(mesh).vertices if orientation else None)
        labels = remove_small_islands(mesh, numbering.labels)
        labels, merged = merge_split_teeth(mesh, labels, jaw, points=orientation.apply(mesh).vertices if orientation else np.asarray(mesh.vertices))
        row["merged"] = merged
        row.update(vertices=len(mesh.vertices), teeth=sorted(int(v) for v in np.unique(labels) if v),
                   reported=[int(m.fdi) for m in detect_missing(mesh, labels, jaw)],
                   warnings=len(check_arch_order(mesh, labels, jaw)))
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return row


def run(zip_path: Path, truth: dict, model: str, out: Path, workers: int, orient: bool, controls: int, limit: int | None, seed: int = 0) -> dict:
    import multiprocessing as mp

    rng = np.random.default_rng(seed)
    cases = [(pid, jaw) for pid, t in truth.items() for jaw in ("upper", "lower") if any(jaw_of(f) == jaw for f in t["missing"])]
    clean = [pid for pid, t in truth.items() if not t["missing"] and not t["other_mentions"] and not t["sentences"]]
    ctrl = [(clean[i], ("upper", "lower")[k % 2]) for k, i in enumerate(rng.permutation(len(clean))[:controls])]
    if limit:
        cases, ctrl = cases[:limit], ctrl[: max(2, limit // 3)]
    names = set(zipfile.ZipFile(zip_path).namelist())
    jobs = [(str(zip_path), pid, jaw, model, orient) for pid, jaw in cases + ctrl if f"{pid}/ios/ios_{jaw}.stl" in names]
    print(f"确认缺牙的颌 {len(cases)} 个，对照 {len(ctrl)} 个", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(_run_one, jobs)):
            rows.append(r)
            if (i + 1) % 20 == 0:
                print(f"[{i + 1}/{len(jobs)}]", flush=True)
    is_ctrl = set(ctrl)
    teeth_rows = []
    for r in rows:
        if "error" in r or (r["patient"], r["jaw"]) in is_ctrl:
            continue
        t = truth[r["patient"]]
        for f in t["missing"]:
            if jaw_of(f) != r["jaw"]:
                continue
            q = f // 10
            k = sum(1 for g in t["missing"] if g // 10 == q)  # 这个象限确认缺了几颗
            in_quadrant = [g for g in r["teeth"] if g // 10 == q and g % 10 <= 7]
            teeth_rows.append({"patient": r["patient"], "fdi": f, "votes": t["votes"][str(f)], "reports": t["reports"],
                               "primary_in_place": f in t["primary_in_place"], "segmented_as_present": f in r["teeth"], "detected": f in r["reported"],
                               # 报了同一象限里相邻的牙位：认出少了一颗，但牙号差一位
                               "detected_or_neighbor": any(g // 10 == q and abs(g % 10 - f % 10) <= 1 for g in r["reported"]),
                               # 这个象限分出来的牙数（不算第三磨牙）没有超过应有的数量：没有把缺牙“补”出来
                               "count_ok": len(in_quadrant) <= 7 - k,
                               "n_teeth": len(r["teeth"]), "reported": r["reported"]})

    def rate(sub, key):
        return round(float(np.mean([x[key] for x in sub])), 3) if sub else None

    def block(sub):
        return {"teeth": len(sub), "detected": rate(sub, "detected"), "detected_or_neighbor": rate(sub, "detected_or_neighbor"),
                "count_ok": rate(sub, "count_ok"), "segmented_as_present": rate(sub, "segmented_as_present")}

    plain = [x for x in teeth_rows if not x["primary_in_place"]]
    kind = lambda f: "molar" if f % 10 >= 6 else "premolar" if f % 10 >= 4 else "anterior"
    c_rows = [r for r in rows if "error" not in r and (r["patient"], r["jaw"]) in is_ctrl]
    summary = {
        "orient": orient,
        "jaws": len(cases), "failed": sum("error" in r for r in rows),
        "confirmed_missing": block(plain),
        "confirmed_by_2plus_reports": block([x for x in plain if x["votes"] >= 2]),
        "by_type": {k: block([x for x in plain if kind(x["fdi"]) == k]) for k in ("anterior", "premolar", "molar")},
        "primary_tooth_in_place": block([x for x in teeth_rows if x["primary_in_place"]]),
        "controls": {"jaws": len(c_rows), "jaws_with_any_reported": sum(bool(r["reported"]) for r in c_rows),
                     "reported_teeth": dict(Counter(f % 10 for r in c_rows for f in r["reported"])),
                     "median_teeth_segmented": float(np.median([len(r["teeth"]) for r in c_rows])) if c_rows else None},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "rows.json").write_text(json.dumps(rows))
    (out / "teeth.json").write_text(json.dumps(teeth_rows))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="真实缺牙测试（Bite2Text）")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--zip", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--zip", type=Path, required=True)
    r.add_argument("--truth", type=Path, required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--no-orient", action="store_true", help="不做朝向归一化（口扫已经是 Teeth3DS 的标准朝向）")
    r.add_argument("--controls", type=int, default=80, help="报告完全没提缺牙的对照颌数")
    r.add_argument("--limit", type=int)
    args = p.parse_args()
    if args.cmd == "extract":
        truth = extract(args.zip)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(truth, indent=1))
        with_m = {k: v for k, v in truth.items() if v["missing"]}
        print(json.dumps({"patients": len(truth), "with_confirmed_missing": len(with_m), "teeth": sum(len(v["missing"]) for v in with_m.values()),
                          "primary_in_place": sum(len(v["primary_in_place"]) for v in with_m.values()),
                          "by_position": dict(sorted(Counter(t % 10 for v in with_m.values() for t in v["missing"]).items()))}))
    else:
        run(args.zip, json.loads(args.truth.read_text()), args.model, args.out, args.workers, not args.no_orient, args.controls, args.limit)


if __name__ == "__main__":
    main()
