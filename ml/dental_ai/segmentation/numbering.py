"""编号纠错：利用牙弓几何自动发现并修正牙号错误。

分割模型最常见的编号错误：
- 缺了一颗牙，后面的牙号整体错位（缺 34 时，35 被认成 34、36 被认成 35……）；
- 左右象限标反（中切牙附近最常见）；
- 两颗相邻的牙被标成同一个号。

做法：
1. 把分割结果拆成牙齿实例（同一牙号、网格上连通的一块；太小的碎片忽略）；
2. 朝向归一化后，沿牙弓从患者右侧最远端到左侧最远端排序；
3. 动态规划：给这串实例分配牙号，牙号沿牙弓严格递增、允许跳过（缺牙），总代价最小。代价由四部分组成：
   - 编号：与模型原来的牙号不一致；
   - 左右：实例在中线一侧，牙号却属于另一侧；
   - 宽度：实例的近远中宽度与该牙位的标准宽度不符（先按整副牙列估计整体比例，消除个体差异和扫描尺度）；
   - 缺隙：相邻两个实例之间跳过了 k 个牙位，两者之间却没有相应宽度的空隙（或反过来）；
4. 只有新方案的代价比原编号低出一个余量，或者原编号本身不合法（重复、顺序错乱）时才修改。

左右的保护：如果大多数远离中线的牙都和标准坐标系里的左右矛盾，更可能是朝向估计反了，
而不是模型把整副牙列标反了。这时按模型的左右翻转坐标后再纠错，并给出警告请人工确认。
所以“整副牙列左右全部标反”这种错误不会被自动纠正，只会提示。

已知不能区分的情况：一侧所有牙号整体错一位、中间又没有缺隙（与“拔牙后间隙已关闭”在几何上几乎一样），保持原样。

只做“重新分配牙号”，不改变分割边界。两颗牙被分成一块、一颗牙被切成两块这类错误，只给出警告。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ..fdi import Jaw, QUADRANTS, arch_order, expected_width, quadrant


@dataclass
class Weights:
    label: float = 0.6  # 与模型原牙号不一致
    mirror: float = 0.3  # 与模型原牙号是左右对称的同名牙（左右混淆，比任意错号便宜）
    side: float = 2.0  # 实例在中线另一侧
    width: float = 2.0  # 宽度不符（相对误差）
    gap: float = 3.0  # 缺隙与跳过的牙位不符（按 7 mm 归一化）
    margin: float = 0.5  # 新方案至少要好这么多才修改
    side_tolerance_mm: float = 3.0  # 离中线这么近的实例不判断左右
    gap_tolerance_mm: float = 2.0


@dataclass
class ToothInstance:
    vertices: np.ndarray  # 顶点下标
    label: int  # 模型给的牙号
    centroid: np.ndarray  # 标准坐标系下的质心
    width: float = 0.0  # 沿牙弓切线方向的宽度（mm）


@dataclass
class NumberingResult:
    labels: np.ndarray
    applied: bool
    changes: list[dict] = field(default_factory=list)  # [{"from": 35, "to": 36, "vertices": 1234}]
    warnings: list[str] = field(default_factory=list)
    cost_before: float | None = None
    cost_after: float | None = None

    def to_dict(self) -> dict:
        return {
            "applied": self.applied,
            "changes": self.changes,
            "warnings": self.warnings,
            "cost_before": None if self.cost_before is None else round(self.cost_before, 3),
            "cost_after": None if self.cost_after is None else round(self.cost_after, 3),
        }


def tooth_instances(mesh: trimesh.Trimesh, labels: np.ndarray, min_fraction: float = 0.2) -> list[np.ndarray]:
    """同一牙号且网格上连通的顶点块。小于“中位数大小 × min_fraction”的碎片忽略。"""
    edges = mesh.edges_unique
    same = (labels[edges[:, 0]] == labels[edges[:, 1]]) & (labels[edges[:, 0]] != 0)
    e = edges[same]
    n = len(labels)
    graph = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    comp = np.where(labels != 0, comp, -1)
    ids, counts = np.unique(comp[comp >= 0], return_counts=True)
    if len(ids) == 0:
        return []
    threshold = max(20, min_fraction * np.median(counts))
    return [np.nonzero(comp == i)[0] for i, c in zip(ids, counts) if c >= threshold]


def _side_of(fdi: int, jaw: Jaw) -> int:
    """-1：患者右侧（标准坐标系 -X），+1：左侧。"""
    right_q, _ = QUADRANTS[jaw]
    return -1 if quadrant(fdi) == right_q else 1


def _measure(instances: list[ToothInstance], points: np.ndarray) -> None:
    """拟合牙弓抛物线 y = a x² + b x + c，按切线方向测每颗牙的宽度。"""
    c = np.array([t.centroid for t in instances])
    if len(c) >= 3:
        a, b, _ = np.polyfit(c[:, 0], c[:, 1], 2)
    else:
        a, b = 0.0, 0.0
    for t in instances:
        x = t.centroid[0]
        tangent = np.array([1.0, 2 * a * x + b])
        tangent /= np.linalg.norm(tangent)
        proj = points[t.vertices][:, :2] @ tangent
        t.width = float(np.ptp(proj))


class _Scorer:
    def __init__(self, instances: list[ToothInstance], points: np.ndarray, jaw: Jaw, w: Weights):
        self.inst = instances
        self.jaw = jaw
        self.w = w
        self.slots = arch_order(jaw, include_third_molars=True)
        # 整体比例：实例宽度 / 原牙号的标准宽度 的中位数，消除个体差异和扫描尺度
        ratios = [t.width / expected_width(t.label) for t in instances if t.label in self.slots and t.width > 0]
        self.scale = float(np.median(ratios)) if ratios else 1.0
        trees = [cKDTree(points[t.vertices]) for t in instances]
        self.gaps = np.zeros(max(len(instances) - 1, 0))
        for i in range(len(instances) - 1):
            d, _ = trees[i + 1].query(points[instances[i].vertices], k=1)
            self.gaps[i] = d.min()

    def unary(self, i: int, k: int) -> float:
        t, w = self.inst[i], self.w
        fdi = self.slots[k]
        cost = 0.0
        if t.label != fdi:
            mirrored = t.label % 10 == fdi % 10 and t.label // 10 != fdi // 10
            cost += w.mirror if mirrored else w.label
        x = t.centroid[0] * _side_of(fdi, self.jaw)
        if x < -w.side_tolerance_mm:
            cost += w.side * min(1.0, (-x - w.side_tolerance_mm) / 5.0)
        expected = expected_width(fdi) * self.scale
        cost += w.width * min(1.0, abs(t.width - expected) / expected)
        return cost

    def pair(self, i: int, j: int, k: int) -> float:
        """实例 i 在牙位 j、实例 i+1 在牙位 k（k > j）之间的缺隙代价。"""
        skipped = sum(expected_width(self.slots[s]) for s in range(j + 1, k)) * self.scale
        miss = max(0.0, abs(self.gaps[i] - skipped) - self.w.gap_tolerance_mm)
        return self.w.gap * min(2.0, miss / 7.0)

    def assignment_cost(self, slots: list[int]) -> float:
        cost = sum(self.unary(i, k) for i, k in enumerate(slots))
        cost += sum(self.pair(i, slots[i], slots[i + 1]) for i in range(len(slots) - 1))
        return cost

    def solve(self) -> tuple[list[int], float]:
        n, m = len(self.inst), len(self.slots)
        dp = np.full((n, m), np.inf)
        back = np.zeros((n, m), dtype=int)
        for k in range(m):
            dp[0, k] = self.unary(0, k)
        for i in range(1, n):
            for k in range(i, m):
                best, arg = np.inf, -1
                for j in range(i - 1, k):
                    if np.isfinite(dp[i - 1, j]):
                        c = dp[i - 1, j] + self.pair(i - 1, j, k)
                        if c < best:
                            best, arg = c, j
                if arg >= 0:
                    dp[i, k] = best + self.unary(i, k)
                    back[i, k] = arg
        k = int(np.argmin(dp[n - 1]))
        cost = float(dp[n - 1, k])
        path = [k]
        for i in range(n - 1, 0, -1):
            k = int(back[i, k])
            path.append(k)
        return path[::-1], cost


def correct_numbering(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    jaw: Jaw,
    canonical_vertices: np.ndarray | None = None,
    weights: Weights | None = None,
) -> NumberingResult:
    """canonical_vertices：标准坐标系下的顶点（+X 患者左侧，+Y 后方）；不提供时内部做一次朝向归一化。"""
    w = weights or Weights()
    labels = np.asarray(labels).copy()
    if canonical_vertices is None:
        from ..orientation import normalize_orientation

        canonical_vertices = normalize_orientation(mesh, jaw).apply(mesh).vertices
    points = np.asarray(canonical_vertices)

    groups = tooth_instances(mesh, labels)
    if len(groups) < 3:
        return NumberingResult(labels, applied=False, warnings=["牙齿太少，跳过编号纠错"])
    slots = arch_order(jaw, include_third_molars=True)
    if len(groups) > len(slots):
        return NumberingResult(labels, applied=False, warnings=[f"识别出 {len(groups)} 块牙齿区域，多于 {len(slots)} 个牙位，可能有过度分割，跳过编号纠错"])

    instances = [
        ToothInstance(vertices=g, label=int(np.bincount(labels[g]).argmax()), centroid=points[g].mean(axis=0)) for g in groups
    ]
    warnings = []
    votes = [
        np.sign(t.centroid[0]) == _side_of(t.label, jaw)
        for t in instances
        if t.label in slots and abs(t.centroid[0]) > w.side_tolerance_mm
    ]
    if votes and np.mean(votes) < 0.4:
        points = points * np.array([-1.0, 1.0, 1.0])
        for t in instances:
            t.centroid = t.centroid * np.array([-1.0, 1.0, 1.0])
        warnings.append("模型的左右与朝向估计相反，已按模型的左右处理，请人工确认左右是否正确")
    instances.sort(key=lambda t: t.centroid[0])  # 患者右侧 → 左侧
    _measure(instances, points)
    scorer = _Scorer(instances, points, jaw, w)

    for t in instances:
        if t.label in slots and t.width > 1.7 * expected_width(t.label) * scorer.scale:
            warnings.append(f"{t.label} 的宽度约为正常的 {t.width / (expected_width(t.label) * scorer.scale):.1f} 倍，可能把两颗牙分成了一颗")

    original = [slots.index(t.label) if t.label in slots else -1 for t in instances]
    valid = all(k >= 0 for k in original) and all(a < b for a, b in zip(original, original[1:]))
    cost_before = scorer.assignment_cost(original) if valid else None

    path, cost_after = scorer.solve()
    apply = cost_before is None or cost_after < cost_before - w.margin
    if not valid:
        warnings.append("原编号不满足沿牙弓递增（有重复或顺序错乱），已重新分配")
    if not apply:
        return NumberingResult(labels, applied=False, warnings=warnings, cost_before=cost_before, cost_after=cost_after)

    changes = []
    out = labels.copy()
    for t, k in zip(instances, path):
        new = slots[k]
        if new != t.label:
            changes.append({"from": t.label, "to": new, "vertices": int(len(t.vertices))})
        out[t.vertices] = new
    return NumberingResult(out, applied=bool(changes), changes=changes, warnings=warnings, cost_before=cost_before, cost_after=cost_after)
