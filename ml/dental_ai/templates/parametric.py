"""内置参数化牙冠模板：按教科书平均尺寸和牙尖分布程序生成，不需要任何数据。

这是**占位用的简化形态**：轮廓是超椭圆，侧面陡峭，咬合面由若干高斯形牙尖叠加而成，
没有真实牙冠的颈部膨隆、窝沟和边缘嵴细节。有了标注数据后，用 `python -m dental_ai.templates build`
从真实牙冠构建模板库替换它。

坐标：牙齿局部坐标系（x 近中、y 颊侧、z 咬合，原点在牙颈线中心），开放网格，底边（z = 0）是牙颈线。
"""

from __future__ import annotations

import numpy as np
import trimesh

from ..fdi import Jaw

# (近远中宽, 颊舌宽, 冠高) mm，常用牙体解剖学教材的平均值
DIMENSIONS: dict[Jaw, dict[int, tuple[float, float, float]]] = {
    "upper": {1: (8.5, 7.0, 10.5), 2: (6.5, 6.0, 9.0), 3: (7.5, 8.0, 10.0), 4: (7.0, 9.0, 8.5),
              5: (6.5, 9.0, 8.5), 6: (10.0, 11.0, 7.5), 7: (9.0, 11.0, 7.0), 8: (8.5, 10.0, 6.5)},
    "lower": {1: (5.0, 6.0, 9.0), 2: (5.5, 6.5, 9.5), 3: (7.0, 7.5, 11.0), 4: (7.0, 7.5, 8.5),
              5: (7.0, 8.0, 8.0), 6: (11.0, 10.5, 7.5), 7: (10.5, 10.0, 7.0), 8: (10.0, 9.5, 7.0)},
}  # fmt: skip


def _cusps(jaw: Jaw, position: int) -> list[tuple[float, float, float, float]]:
    """牙尖：(x, y 相对半轴的位置, 高度占冠高的比例, 宽度占近远中半宽的比例)。x 近中为正，y 颊侧为正。"""
    if position in (1, 2):  # 切牙：一条切嵴，用几个靠近唇侧的小尖近似
        return [(x, 0.15, 0.22, 0.35) for x in (-0.55, 0.0, 0.55)]
    if position == 3:  # 尖牙：一个尖
        return [(0.05, 0.1, 0.32, 0.45)]
    if position in (4, 5):  # 前磨牙：颊尖高、舌尖低
        lingual = 0.08 if (jaw == "lower" and position == 4) else 0.16
        return [(0.0, 0.42, 0.26, 0.4), (0.0, -0.42, lingual, 0.35)]
    if jaw == "lower" and position == 6:  # 下颌第一磨牙：5 个尖
        return [(0.5, 0.45, 0.2, 0.3), (-0.05, 0.5, 0.19, 0.3), (-0.6, 0.35, 0.15, 0.25),
                (0.45, -0.45, 0.22, 0.3), (-0.35, -0.45, 0.2, 0.3)]  # fmt: skip
    # 其它磨牙：4 个尖（上颌近中舌尖最大）
    ml = 0.24 if jaw == "upper" else 0.21
    return [(0.45, 0.45, 0.2, 0.32), (-0.4, 0.45, 0.18, 0.3), (0.4, -0.45, ml, 0.35), (-0.4, -0.45, 0.16, 0.28)]


def parametric_crown(jaw: Jaw, position: int, rings: int = 24, sectors: int = 72) -> trimesh.Trimesh:
    md, bl, height = DIMENSIONS[jaw][position]
    a, c = md / 2, bl / 2
    square = 2.6 if position >= 6 else 2.2  # 磨牙轮廓更方
    body = 0.75 * height

    rho = np.linspace(0, 1, rings + 1)[1:]
    phi = np.linspace(0, 2 * np.pi, sectors, endpoint=False)
    R, F = np.meshgrid(rho, phi, indexing="ij")
    cos, sin = np.cos(F), np.sin(F)
    # 超椭圆轮廓
    x = a * R * np.sign(cos) * np.abs(cos) ** (2 / square)
    y = c * R * np.sign(sin) * np.abs(sin) ** (2 / square)

    def surface(x, y, r):
        z = body * (1 - r**4) ** 0.25  # 侧面陡峭、顶部平缓
        fade = np.clip(1 - r**2, 0, 1)
        for cx, cy, amp, width in _cusps(jaw, position):
            sigma = width * a
            z = z + amp * height * np.exp(-((x - cx * a) ** 2 + (y - cy * c) ** 2) / (2 * sigma**2)) * fade
        if position >= 6:  # 中央窝
            z = z - 0.08 * height * np.exp(-(x**2 + y**2) / (2 * (0.3 * a) ** 2))
        return z

    Z = surface(x, y, R)
    Z[-1, :] = 0.0  # 底边（牙颈线）严格在 z = 0
    top = surface(np.zeros(1), np.zeros(1), np.zeros(1))[0]
    verts = np.vstack([[0.0, 0.0, top], np.stack([x.ravel(), y.ravel(), Z.ravel()], axis=1)])

    def vid(i, j):
        return 1 + i * sectors + (j % sectors)

    faces = [[0, vid(0, j), vid(0, j + 1)] for j in range(sectors)]
    for i in range(rings - 1):
        for j in range(sectors):
            a0, a1, b0, b1 = vid(i, j), vid(i, j + 1), vid(i + 1, j), vid(i + 1, j + 1)
            faces += [[a0, b0, b1], [a0, b1, a1]]
    mesh = trimesh.Trimesh(verts, np.array(faces), process=False)
    if (mesh.face_normals[:, 2] * mesh.area_faces).sum() < 0:
        mesh.invert()
    return mesh
