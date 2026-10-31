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
两颗相邻的牙被标成同一个号时它们是连通的一块，宽度约为两倍：动态规划允许这样的实例占两个相邻牙位，
它自己的牙号不动（等人工在编辑器里拆开），但两侧的牙不会因为“少了一颗牙”而被错误地整体移位。
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
    gap: float = 3.0  # 有缺隙却没有跳过牙位（按 7 mm 归一化）：缺牙后整体错位的典型表现
    closed_gap: float = 0.3  # 跳过了牙位却没有缺隙，相对 gap 的比例：正畸拔牙后间隙关闭很常见（真实数据上），只轻罚
    margin: float = 0.5  # 新方案至少要好这么多才修改
    merged: float = 1.0  # 一个实例占两个牙位（两颗牙连成一块）的额外代价
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


def _arch_fit(instances: list[ToothInstance]) -> np.ndarray:
    c = np.array([t.centroid for t in instances])
    return np.polyfit(c[:, 0], c[:, 1], 2) if len(c) >= 3 else np.array([0.0, 0.0, float(c[:, 1].mean())])


def _arch_position(instances: list[ToothInstance], coef: np.ndarray) -> np.ndarray:
    """每颗牙在牙弓上的位置：质心投影到拟合的牙弓抛物线上，取投影点的 x（沿曲线单调）。

    直接按质心 x 排序不行：真实牙弓上拥挤、颊侧错位的牙（常见是尖牙），以及后牙区牙弓接近平行于 y 轴的地方，
    相邻两颗牙的质心 x 会颠倒（Teeth3DS 测试集上 13% 的正确编号因此被误改）。"""
    xs = np.linspace(-60, 60, 4001)
    curve = np.stack([xs, np.polyval(coef, xs)], axis=1)
    tree = cKDTree(curve)
    return np.array([xs[tree.query(t.centroid[:2])[1]] for t in instances])


def _measure(instances: list[ToothInstance], points: np.ndarray, coef: np.ndarray) -> None:
    """按牙弓抛物线 y = a x² + b x + c 的切线方向测每颗牙的宽度。"""
    a, b, _ = coef
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

    def unary_merged(self, i: int, k: int) -> float:
        """实例 i 同时占牙位 k-1 和 k（两颗牙被标成同一个号、连成一块）。"""
        t, w = self.inst[i], self.w
        a, b = self.slots[k - 1], self.slots[k]
        cost = w.merged + (0.0 if t.label in (a, b) else w.label)
        for fdi in (a, b):
            x = t.centroid[0] * _side_of(fdi, self.jaw)
            if x < -w.side_tolerance_mm - 4.0:  # 跨中线的一对（41 + 31）质心就在中线附近
                cost += w.side * min(1.0, (-x - w.side_tolerance_mm - 4.0) / 5.0)
        expected = (expected_width(a) + expected_width(b)) * self.scale
        cost += w.width * min(1.0, abs(t.width - expected) / expected)
        return cost

    def pair(self, i: int, j: int, k: int) -> float:
        """实例 i 结束于牙位 j、实例 i+1 开始于牙位 k（k > j）之间的缺隙代价。"""
        skipped = sum(expected_width(self.slots[s]) for s in range(j + 1, k)) * self.scale
        diff = self.gaps[i] - skipped
        miss = max(0.0, abs(diff) - self.w.gap_tolerance_mm)
        weight = self.w.gap if diff > 0 else self.w.gap * self.w.closed_gap
        return weight * min(2.0, miss / 7.0)

    def assignment_cost(self, slots: list[int]) -> float:
        cost = sum(self.unary(i, k) for i, k in enumerate(slots))
        cost += sum(self.pair(i, slots[i], slots[i + 1]) for i in range(len(slots) - 1))
        return cost

    def solve(self) -> tuple[list[tuple[int, int]], float]:
        """返回每个实例占的牙位区间 (起, 止)（下标；起 == 止 是一颗牙，止 == 起 + 1 是连成一块的两颗牙）和总代价。"""
        n, m = len(self.inst), len(self.slots)
        # 状态：(实例 i, 结束牙位 k, 是否占两个牙位)
        dp = np.full((n, m, 2), np.inf)
        back = np.zeros((n, m, 2, 2), dtype=int)
        for i in range(n):
            for k in range(m):
                for merged in (0, 1):
                    start = k - merged
                    if start < 0:
                        continue
                    own = self.unary_merged(i, k) if merged else self.unary(i, k)
                    if i == 0:
                        dp[i, k, merged] = own
                        continue
                    best, arg = np.inf, (-1, 0)
                    for j in range(start):
                        for pm in (0, 1):
                            if np.isfinite(dp[i - 1, j, pm]):
                                c = dp[i - 1, j, pm] + self.pair(i - 1, j, start)
                                if c < best:
                                    best, arg = c, (j, pm)
                    if arg[0] >= 0:
                        dp[i, k, merged] = best + own
                        back[i, k, merged] = arg
        k, merged = np.unravel_index(int(np.argmin(dp[n - 1])), (m, 2))
        cost = float(dp[n - 1, k, merged])
        path = []
        for i in range(n - 1, -1, -1):
            path.append((int(k - merged), int(k)))
            k, merged = back[i, k, merged]
        return path[::-1], cost


def _split_merged(t: ToothInstance, points: np.ndarray, coef: np.ndarray, first: int, second: int) -> tuple[np.ndarray, np.ndarray]:
    """把占两个牙位的一块沿牙弓切成两块，返回 (沿牙弓在前的部分, 在后的部分)。

    沿牙弓切线把顶点投影成一维；切点先按两颗牙的标准宽度比例定，再在附近（±20% 宽度）找牙冠顶面轮廓最低的位置——
    两颗牙之间的邻间隙。"""
    p = points[t.vertices]
    x = t.centroid[0]
    tangent = np.array([1.0, 2 * coef[0] * x + coef[1]])
    tangent /= np.linalg.norm(tangent)
    u = p[:, :2] @ tangent
    lo, hi = np.percentile(u, 1), np.percentile(u, 99)
    w1, w2 = expected_width(first), expected_width(second)
    guess = lo + (hi - lo) * w1 / (w1 + w2)
    bins = np.linspace(lo, hi, 61)
    idx = np.clip(np.digitize(u, bins) - 1, 0, len(bins) - 2)
    top = np.full(len(bins) - 1, -np.inf)
    np.maximum.at(top, idx, p[:, 2])  # 每个位置的牙冠最高点（标准坐标系牙冠朝 +Z）
    centers = (bins[:-1] + bins[1:]) / 2
    window = np.abs(centers - guess) <= 0.2 * (hi - lo)
    window &= np.isfinite(top)
    cut = centers[window][np.argmin(top[window])] if window.any() else guess
    return t.vertices[u < cut], t.vertices[u >= cut]


def correct_numbering(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    jaw: Jaw,
    canonical_vertices: np.ndarray | None = None,
    weights: Weights | None = None,
    split_merged: bool = True,
) -> NumberingResult:
    """split_merged：两颗牙连成一块时自动拆开（否则只提示）。
    canonical_vertices：标准坐标系下的顶点（见 orientation.py：+Y 后方，下颌 +X 为患者左侧、上颌为患者右侧）；
    不提供时内部做一次朝向归一化。"""
    w = weights or Weights()
    labels = np.asarray(labels).copy()
    if canonical_vertices is None:
        from ..orientation import normalize_orientation

        canonical_vertices = normalize_orientation(mesh, jaw).apply(mesh).vertices
    points = np.asarray(canonical_vertices)
    if jaw == "upper":
        # 标准坐标系里上颌的患者左侧在 -X（见 orientation.py），这里统一成“患者左侧 = +X”
        points = points * np.array([-1.0, 1.0, 1.0])

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
    coef = _arch_fit(instances)
    position = _arch_position(instances, coef)
    instances = [instances[i] for i in np.argsort(position)]  # 沿牙弓，患者右侧 → 左侧
    _measure(instances, points, coef)
    scorer = _Scorer(instances, points, jaw, w)

    for t in instances:
        if t.label in slots and t.width > 1.7 * expected_width(t.label) * scorer.scale:
            warnings.append(f"{t.label} 的宽度约为正常的 {t.width / (expected_width(t.label) * scorer.scale):.1f} 倍，可能是两颗牙连成了一块")

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
    for t, (start, end) in zip(instances, path):
        if start != end and split_merged:
            first, second = _split_merged(t, points, coef, slots[start], slots[end])
            for part, fdi in ((first, slots[start]), (second, slots[end])):
                out[part] = fdi
                if fdi != t.label:
                    changes.append({"from": t.label, "to": fdi, "vertices": int(len(part))})
            pair = tuple(sorted((slots[start], slots[end])))
            warnings.append(f"{t.label} 这一块应该是 {pair[0]} 和 {pair[1]} 两颗牙，已在牙间最低处自动拆开，请在编辑器里确认边界")
            continue
        if start != end:
            pair = tuple(sorted((slots[start], slots[end])))
            new = t.label if t.label in pair else slots[start]
            warnings.append(f"{t.label} 这一块应该是 {pair[0]} 和 {pair[1]} 两颗牙，请在编辑器里用画笔拆开")
        else:
            new = slots[start]
        if new != t.label:
            changes.append({"from": t.label, "to": new, "vertices": int(len(t.vertices))})
        out[t.vertices] = new
    # 没有被任何面引用的孤立顶点（真实 Teeth3DS 扫描里约 7%）不属于任何实例，跟随最近的被引用顶点
    referenced = np.zeros(len(labels), dtype=bool)
    referenced[mesh.faces] = True
    isolated = np.nonzero(~referenced & (labels != 0))[0]
    if changes and len(isolated):
        ref = np.nonzero(referenced)[0]
        _, nearest = cKDTree(np.asarray(mesh.vertices)[ref]).query(np.asarray(mesh.vertices)[isolated])
        out[isolated] = out[ref[nearest]]
    return NumberingResult(out, applied=bool(changes), changes=changes, warnings=warnings, cost_before=cost_before, cost_after=cost_after)
