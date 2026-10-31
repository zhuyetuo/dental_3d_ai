"""从下载的压缩包里导出数据集，统一成 Teeth3DS 目录结构：<out>/<病例>/<病例>_<lower|upper>.obj + .json（逐顶点 FDI）。

解压全部 Teeth3DS + Poseidon3D 要 40 GB，这里只按划分导出需要的病例。导出后评估、模板库、流程、网页都能直接用。

    python -m dental_ai.datasets teeth3ds --zips data/drive/raw/Teeth3DS --split <划分文件> --out data/teeth3ds_test --limit 100
    python -m dental_ai.datasets poseidon3d --zip data/drive/raw/Poseidon3D/poseidon3d.zip --split test-complex --out data/poseidon3d

Poseidon3D（Kubík & Španěl 2024）的格式（官方没有说明，从文件内容推出来的）：
- data/<病例>/<病例>_MODEL_mandible|maxilla.stl：一个颌（mandible 下颌、maxilla 上颌）；
- <病例>_REGIONS_FACES_<颌>.pkl：逐三角形的区域号（0, 1, 2…，17 为牙龈），顺序与 STL 的三角形一致；
- <病例>_MAPPING_<颌>.pkl：17 个位置，位置 p（1–16）沿牙弓从患者右侧到左侧，值是该位置对应的区域号（-1 为缺失）。
  p = 1…8 → 48…41（上颌 18…11），p = 9…16 → 31…38（上颌 21…28）。这个对应关系是用同时带 FDI 牙号的
  *_QUADRANT_NUMBERING_* 文件（只有部分病例有）核对出来的；
- data/split.txt：test-simple、test-complex（拥挤、缺牙等困难病例）、validation、train；
- 模型是带底座的封闭实体（不是开放的口扫曲面），左右手性与 Teeth3DS 相反，导出时镜像（见 load_poseidon3d_case）。
pkl 是 Python pickle，加载不可信的 pickle 可能执行任意代码；这些文件只含元组、列表、数字，
用禁止加载任何类的 Unpickler 读取。
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from .io import save_labels

JAWS = {"mandible": "lower", "maxilla": "upper"}


class _PlainUnpickler(pickle.Unpickler):
    def find_class(self, module, name):  # 只允许基本类型
        raise pickle.UnpicklingError(f"拒绝加载 {module}.{name}")


def safe_unpickle(data: bytes):
    return _PlainUnpickler(io.BytesIO(data)).load()


def face_to_vertex_labels(faces: np.ndarray, face_labels: np.ndarray, n_vertices: int) -> np.ndarray:
    """顶点标签 = 相邻三角形标签的众数（平票取牙号较大的，牙齿优先于牙龈）。"""
    values = np.unique(face_labels)
    counts = np.zeros((n_vertices, len(values)), dtype=np.int32)
    idx = np.searchsorted(values, face_labels)
    for k in range(3):
        np.add.at(counts, (faces[:, k], idx), 1)
    best = counts.shape[1] - 1 - np.argmax(counts[:, ::-1], axis=1)
    return values[best].astype(np.int64)


def teeth3ds_members(zip_dir: Path) -> dict[str, dict[str, tuple[Path, str]]]:
    """{扫描名: {"obj": (zip, 成员名), "json": ..., "kpt": ...}}。关键点在 3DTeethLand_landmarks_annotation/*.zip。"""
    members: dict[str, dict[str, tuple[Path, str]]] = {}
    for z in sorted(zip_dir.glob("data_part_*.zip")) + sorted(zip_dir.glob("3DTeethLand_landmarks_annotation/*.zip")):
        for name in zipfile.ZipFile(z).namelist():
            stem = Path(name).stem
            if name.endswith("__kpt.json"):
                members.setdefault(stem[: -len("__kpt")], {})["kpt"] = (z, name)
            elif name.endswith((".obj", ".json")):
                members.setdefault(stem, {})[name.rsplit(".", 1)[-1]] = (z, name)
    return members


@dataclass
class ZipScan:
    """压缩包里的一个扫描，load() 时才读（可以传给多进程）。"""

    name: str
    jaw: str
    obj: tuple[Path, str]
    labels: tuple[Path, str]

    def load(self) -> tuple[trimesh.Trimesh, np.ndarray]:
        from .io import parse_obj

        with zipfile.ZipFile(self.obj[0]) as zf:
            mesh = parse_obj(zf.read(self.obj[1]).decode("utf-8", errors="replace").splitlines())
        with zipfile.ZipFile(self.labels[0]) as zf:
            labels = np.asarray(json.loads(zf.read(self.labels[1]))["labels"], dtype=np.int64)
        return mesh, labels


def teeth3ds_zip_scans(zip_dir: Path, split: Path | None = None) -> list[ZipScan]:
    members = teeth3ds_members(zip_dir)
    wanted = [line.strip() for line in split.read_text().splitlines() if line.strip()] if split else sorted(members)
    return [ZipScan(s, s.rsplit("_", 1)[1], members[s]["obj"], members[s]["json"])
            for s in wanted if "obj" in members.get(s, {}) and "json" in members[s]]


def iter_teeth3ds(zip_dir: Path, stems: list[str] | None = None, members=None):
    """直接从压缩包读（不解压到磁盘）：逐个产出 (扫描名, 颌位, 网格, 逐顶点标签, 关键点 JSON 或 None)。"""
    from .io import parse_obj

    members = members or teeth3ds_members(zip_dir)
    for stem in stems or sorted(members):
        m = members.get(stem, {})
        if "obj" not in m or "json" not in m:
            continue
        with zipfile.ZipFile(m["obj"][0]) as zf:
            mesh = parse_obj(zf.read(m["obj"][1]).decode("utf-8", errors="replace").splitlines())
            labels = np.asarray(json.loads(zf.read(m["json"][1]))["labels"], dtype=np.int64)
        kpt = None
        if "kpt" in m:
            with zipfile.ZipFile(m["kpt"][0]) as zf:
                kpt = json.loads(zf.read(m["kpt"][1]))
        yield stem, stem.rsplit("_", 1)[1], mesh, labels, kpt


def export_teeth3ds(zip_dir: Path, out: Path, split: Path | None = None, limit: int | None = None) -> list[str]:
    """按划分把扫描（OBJ + 标注 JSON，有关键点时连同 __kpt.json）从压缩包拷出来。"""
    members = teeth3ds_members(zip_dir)
    wanted = [line.strip() for line in split.read_text().splitlines() if line.strip()] if split else sorted(members)
    stems = [s for s in wanted if "obj" in members.get(s, {}) and "json" in members[s]]
    if limit:
        stems = stems[:limit]
    for stem in stems:
        case = stem.rsplit("_", 1)[0]
        for z, name in members[stem].values():
            dest = out / case / Path(name).name
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(z) as zf, zf.open(name) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
    return stems


def poseidon3d_split(zf: zipfile.ZipFile) -> dict[str, list[str]]:
    splits = {}
    for line in zf.read("data/split.txt").decode().splitlines():
        if ":" in line:
            name, cases = line.split(":", 1)
            splits[name.strip()] = [c.strip() for c in cases.split(",") if c.strip()]
    return splits


def poseidon3d_fdi(regions, mapping, jaw: str) -> np.ndarray:
    """区域号 → FDI。mapping[p] = 位置 p 的区域号；位置 1–8 是患者右侧（从远中到近中），9–16 是左侧。"""
    right, left = (4, 3) if jaw == "lower" else (1, 2)
    lut = {}
    for p, region in enumerate(mapping):
        if region is None or region < 0 or not 1 <= p <= 16:
            continue
        lut[region] = right * 10 + (9 - p) if p <= 8 else left * 10 + (p - 8)
    regions = np.asarray(regions, dtype=np.int64)
    return np.array([lut.get(r, 0) for r in regions.tolist()], dtype=np.int64)


def load_poseidon3d_case(zf: zipfile.ZipFile, case: str) -> list[tuple[str, trimesh.Trimesh, np.ndarray]]:
    """返回 [(颌位, 网格, 逐顶点 FDI)]。"""
    names = [n for n in zf.namelist() if n.startswith(f"data/{case}/")]
    out = []
    for model in [n for n in names if "_MODEL_" in n and n.endswith(".stl")]:
        part = model.rsplit("_", 1)[-1][:-4]  # mandible / maxilla
        regions_file = f"data/{case}/{case}_REGIONS_FACES_{part}.pkl"
        mapping_file = f"data/{case}/{case}_MAPPING_{part}.pkl"
        if regions_file not in names or mapping_file not in names:
            continue
        mesh = trimesh.load(io.BytesIO(zf.read(model)), file_type="stl", process=False)
        face_labels = poseidon3d_fdi(safe_unpickle(zf.read(regions_file)), safe_unpickle(zf.read(mapping_file)), JAWS[part])
        if len(face_labels) != len(mesh.faces):
            raise ValueError(f"{case} {part}: 标签数 {len(face_labels)} 与三角形数 {len(mesh.faces)} 不一致")
        mesh.merge_vertices()  # STL 没有共享顶点；合并后三角形顺序不变
        # 镜像：Poseidon3D 的模型与标注的左右手性和 Teeth3DS（符合解剖）相反——按右手系、牙冠朝上、
        # 磨牙在后摆正后，标成“患者左侧”的牙在患者右侧（测试集 40 个病例全部如此）。单看形状分不出是模型
        # 被镜像了还是左右标反了；统一成和 Teeth3DS 一致，同一个牙号在两个数据集里才在同一侧
        mesh.vertices = mesh.vertices * np.array([-1.0, 1.0, 1.0])
        mesh.invert()
        labels = face_to_vertex_labels(mesh.faces, face_labels, len(mesh.vertices))
        out.append((JAWS[part], mesh, labels))
    return out


def export_poseidon3d(zip_path: Path, out: Path, split: str | None = None, limit: int | None = None) -> list[str]:
    zf = zipfile.ZipFile(zip_path)
    splits = poseidon3d_split(zf)
    if split:
        if split not in splits:
            raise SystemExit(f"划分 {split} 不存在，可选：{', '.join(splits)}")
        cases = splits[split]
    else:
        cases = sorted({n.split("/")[1] for n in zf.namelist() if n.startswith("data/") and n.count("/") >= 2 and n.split("/")[1]})
    if limit:
        cases = cases[:limit]
    done = []
    for case in cases:
        for jaw, mesh, labels in load_poseidon3d_case(zf, case):
            stem = f"P{case}_{jaw}"
            dest = out / f"P{case}" / f"{stem}.obj"
            dest.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(dest)
            save_labels(dest.with_suffix(".json"), labels, jaw, patient_id=f"poseidon3d:{case}")
            done.append(stem)
    return done


def main() -> None:
    p = argparse.ArgumentParser(description="从压缩包导出数据集（Teeth3DS 目录结构）")
    sub = p.add_subparsers(dest="dataset", required=True)
    t = sub.add_parser("teeth3ds")
    t.add_argument("--zips", type=Path, required=True, help="data_part_*.zip 所在目录")
    t.add_argument("--split", type=Path, help="划分文件（每行 <病例>_<颌位>）")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--limit", type=int)
    q = sub.add_parser("poseidon3d")
    q.add_argument("--zip", type=Path, required=True)
    q.add_argument("--split", help="test-simple / test-complex / validation / train")
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--limit", type=int)
    args = p.parse_args()
    if args.dataset == "teeth3ds":
        done = export_teeth3ds(args.zips, args.out, args.split, args.limit)
    else:
        done = export_poseidon3d(args.zip, args.out, args.split, args.limit)
    print(f"导出 {len(done)} 个扫描到 {args.out}")


if __name__ == "__main__":
    main()
