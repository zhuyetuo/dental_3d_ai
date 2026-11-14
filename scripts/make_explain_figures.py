"""“先看懂指标”的卡通示意图：统一画法生成 SVG，替换三个专项页（分割、牙冠生成、颈缘线）里的旧图。幂等（按标记整块替换）。"""
import re
from pathlib import Path

D = Path.home() / "zyt/dental_3d_ai/docs"

IVORY = ("#fffaf0", "#e7d8b0", "#b39d6b")      # 天然牙：亮、暗、描边
CYAN = ("#9ff3fa", "#1fb9c9", "#0b7f8c")       # 生成的牙
GRAY = ("#e3e9f5", "#a9b6cf", "#6f7f9e")       # 对颌
GUM = ("#f7c1b9", "#e38f86")
GOOD, BAD, WARN, TXT, DIM = "#3ddc97", "#ff6b6b", "#f5b74f", "#e8eef8", "#9aa8c4"
_uid = [0]


def uid(p):
    _uid[0] += 1
    return f"{p}{_uid[0]}"


FLAT = {IVORY: "#f1e4c1", CYAN: "#38c4d1", GRAY: "#b9c5dc", GUM: "#ee9f95"}  # 扁平风格：每种东西一个纯色


def grad(gid, c, vertical=True):
    """扁平风格：纯色（保留 id 写法，调用处不用改）。"""
    flat = FLAT.get(tuple(c), c[1] if len(c) > 1 else c[0])
    return f'<linearGradient id="{gid}"><stop offset="0" stop-color="{flat}"/><stop offset="1" stop-color="{flat}"/></linearGradient>'


def molar(x, y, w, h, colors, flip=False, extra="", dash=False):
    """卡通磨牙冠：顶上三个牙尖、圆润的侧面、牙颈略收。flip=True 画成对颌（上面倒挂）。"""
    g = uid("g")
    L, R, T, B = x, x + w, y, y + h
    d = (f"M{L + .1 * w:.1f},{B:.1f} C{L - .04 * w:.1f},{T + .62 * h:.1f} {L - .02 * w:.1f},{T + .22 * h:.1f} {L + .16 * w:.1f},{T + .1 * h:.1f} "
         f"Q{L + .27 * w:.1f},{T - .04 * h:.1f} {L + .37 * w:.1f},{T + .1 * h:.1f} Q{L + .5 * w:.1f},{T + .24 * h:.1f} {L + .63 * w:.1f},{T + .1 * h:.1f} "
         f"Q{L + .73 * w:.1f},{T - .04 * h:.1f} {L + .84 * w:.1f},{T + .1 * h:.1f} C{R + .02 * w:.1f},{T + .22 * h:.1f} {R + .04 * w:.1f},{T + .62 * h:.1f} {R - .1 * w:.1f},{B:.1f} "
         f"Q{L + .5 * w:.1f},{B + .07 * h:.1f} {L + .1 * w:.1f},{B:.1f} Z")
    tr = f' transform="translate(0,{2 * y + h:.1f}) scale(1,-1)"' if flip else ""
    outline = 'fill="none"' in extra
    extra = extra.replace('fill="none"', "")
    width = "4" if outline else "2.5"
    stroke = f'stroke="{colors[2] if not outline else colors[1]}" stroke-width="{width}"' + (' stroke-dasharray="9 6"' if dash else "")
    extra = re.sub(r'stroke-width="[^"]*"', "", extra)
    hl = ""  # 扁平风格：不画高光
    fill = "none" if outline else f"url(#{g})"
    return f'<defs>{grad(g, colors)}</defs><g{tr}><path d="{d}" fill="{fill}" {stroke} {extra}/>{hl}</g>'


def gum(x, y, w, h=40):
    g = uid("gum")
    waves = " ".join(f"Q{x + (i + .5) * w / 6:.1f},{y - 10:.1f} {x + (i + 1) * w / 6:.1f},{y:.1f}" for i in range(6))
    return f'<defs>{grad(g, GUM)}</defs><path d="M{x},{y + h} L{x},{y} {waves} L{x + w},{y + h} Z" fill="url(#{g})"/>'


def badge(x, y, text, color, dark=False, size=15):
    w = 14 + len(text) * size * 0.95
    fg = "#08131f" if not dark else TXT
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{size + 12}" rx="{(size + 12) / 2}" fill="{color}"/>'
            f'<text x="{x + w / 2:.1f}" y="{y + size + 4:.1f}" text-anchor="middle" font-size="{size}" font-weight="700" fill="{fg}">{text}</text>')


def label(x, y, text, color=TXT, size=16, anchor="start", weight="600"):
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{text}</text>'


def svg(w, h, body, aria):
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{aria}" font-family="Inter, PingFang SC, Microsoft YaHei, sans-serif">{body}</svg>'


def fig(title, caption, s, wide=False):
    style = ' style="grid-column:1/-1"' if wide else ""
    return f'      <div class="fig"{style}>\n        {s}\n        <b>{title}</b><span>{caption}</span>\n      </div>\n'


def views_fig(title, caption, img, src):
    """真实数据的图：静态图 / 动图 / 3D 交互 三个按钮切换（views.js）。"""
    return f"""      <div class="fig" style="grid-column:1/-1">
        <div class="views" data-src="{src}" data-active="static">
          <div class="vbar"><button type="button" class="vbtn on" data-view="static">静态图</button><button type="button" class="vbtn" data-view="anim">动图</button><button type="button" class="vbtn" data-view="3d">3D 交互</button></div>
          <div class="vpane" data-view="static"><img src="{img}" alt="{title}" loading="lazy"></div>
          <div class="vpane" data-view="anim" hidden><canvas></canvas></div>
          <div class="vpane" data-view="3d" hidden><div class="v3d"></div>
            <div class="vctl"><label>离对颌 <input type="range" min="-0.5" max="1" step="0.02" value="0.6"></label><output></output>
            <button type="button" class="vopa">对颌：半透明 / 不透明</button><span class="vhint">拖动旋转 · 滚轮缩放 · 右键平移</span></div></div>
        </div>
        <b>{title}</b><span>{caption}</span>
      </div>
"""


def qpt(p0, p1, p2, t):
    """二次贝塞尔曲线上参数 t 的点。"""
    return tuple((1 - t) ** 2 * a + 2 * (1 - t) * t * b + t ** 2 * c for a, b, c in zip(p0, p1, p2))


def panel(x, title, ok, body):
    """一个小场景：上面写结论（✓ / ✗），下面画图。"""
    c = GOOD if ok is True else BAD if ok is False else WARN
    mark = "✓" if ok is True else "✗" if ok is False else "!"
    return (f'<g transform="translate({x},0)"><rect x="4" y="4" width="182" height="252" rx="16" fill="#0b1426" stroke="{c}" stroke-opacity=".55" stroke-width="2"/>'
            f'<circle cx="30" cy="32" r="14" fill="{c}"/><text x="30" y="38" text-anchor="middle" font-size="17" font-weight="800" fill="#08131f">{mark}</text>'
            f'{label(52, 38, title, c, 16, weight="700")}{body}</g>')


# ---------------- 牙冠生成 ----------------
def gen_figs():
    out = []
    # 1. 形状误差
    b = molar(60, 96, 280, 150, IVORY)
    b += molar(80, 76, 290, 164, CYAN, dash=True, extra='fill="none" stroke-width="4"')
    for x0, y0, x1, y1 in ((140, 98, 154, 80), (268, 98, 284, 80), (57, 165, 77, 165), (59, 200, 81, 200), (343, 165, 368, 165)):
        b += f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="#ffffff" stroke-width="3.5" stroke-linecap="round"/>'
    b += f'<line x1="330" y1="236" x2="362" y2="236" stroke="{BAD}" stroke-width="5" stroke-linecap="round"/><circle cx="362" cy="236" r="6" fill="{BAD}"/>'
    b += badge(20, 18, "真牙", IVORY[1]) + badge(96, 18, "生成的牙（虚线）", CYAN[1])
    b += label(378, 222, "最远的一处", BAD, 16) + label(378, 246, "= Hausdorff", BAD, 16)
    b += label(20, 285, "白线的平均长度 = Chamfer（整体差多少）", TXT, 16)
    out.append(fig("形状误差", "生成的牙和被挖掉的真牙表面差多远：Chamfer 看平均，Hausdorff 看最差的那一处。越小越好。", svg(520, 300, b, "形状误差")))

    # 2. 和对颌：三种情况
    def bite(gap, pen):
        s = molar(30, 40 + gap, 130, 70, GRAY, flip=True)
        s += molar(30, 128, 130, 90, CYAN)
        if pen:
            s += f'<ellipse cx="95" cy="128" rx="26" ry="9" fill="{BAD}" fill-opacity=".85"/>'
        return s + gum(4, 210, 182, 46)
    b = panel(0, "没咬上", False, bite(-14, False) + f'<line x1="95" y1="100" x2="95" y2="128" stroke="{WARN}" stroke-width="3" stroke-dasharray="4 3"/>' + label(104, 120, "空", WARN, 15))
    b += panel(196, "刚好接触", True, bite(0, False) + f'<circle cx="74" cy="128" r="6" fill="{GOOD}"/><circle cx="118" cy="128" r="6" fill="{GOOD}"/>')
    b += panel(392, "穿进对颌", False, bite(14, True))
    out.append(views_fig("和对颌（上面的牙）",
                         "真实剖面（Bite2Text 一颗下颌磨牙和它的对颌）：要刚好接触，空着咬不到东西，穿进去真做出来要磨或者咬不下去。"
                         "点“动图”看牙冠上下移动，点“3D 交互”可以拖动旋转、用滑块调高低（颜色：蓝 = 有间隙，绿 = 刚好接触，红 = 穿进去）。",
                         "images/explain_bite.png", "assets/bite/F5075_36"))

    # 3. 和邻牙：三种情况
    def contact(shift, tag):
        s = molar(8, 92, 56, 110, IVORY) + molar(126, 92, 56, 110, IVORY)
        s += molar(62 - shift, 86, 66 + 2 * shift, 116, CYAN)
        return s + gum(4, 200, 182, 56) + tag
    b = panel(0, "有缝", False, contact(-8, f'<path d="M58 140 l8 0" stroke="{WARN}" stroke-width="3"/>' + label(40, 238, "塞食物", "#5a1d18", 15)))
    b += panel(196, "刚好贴上", True, contact(0, f'<circle cx="64" cy="128" r="6" fill="{GOOD}"/><circle cx="126" cy="128" r="6" fill="{GOOD}"/>'))
    b += panel(392, "挤进邻牙", False, contact(9, f'<ellipse cx="62" cy="140" rx="7" ry="16" fill="{BAD}"/><ellipse cx="128" cy="140" rx="7" ry="16" fill="{BAD}"/>' + label(36, 238, "戴不进去", "#5a1d18", 15)))
    out.append(fig("和左右邻牙", "米色：邻牙；青：生成的牙。两边都要刚好贴上：有缝会塞食物，挤进去牙冠戴不进去。", svg(582, 262, b, "邻接三种情况"), wide=True))

    # 4. 牙尖和中央窝
    g = uid("g")
    b = f'<defs>{grad(g, CYAN)}</defs>'
    b += (f'<path d="M30 250 L30 120 Q80 30 135 70 Q175 105 200 175 Q225 210 260 172 Q300 110 345 60 Q400 30 450 120 L450 250 Z" '
          f'fill="url(#{g})" stroke="{CYAN[2]}" stroke-width="3"/>')
    for cx, cy in ((95, 48), (380, 46)):
        b += f'<path d="M{cx - 16},{cy + 4} L{cx},{cy - 18} L{cx + 16},{cy + 4} Z" fill="{WARN}"/>'
    b += f'<circle cx="232" cy="196" r="10" fill="{GOOD}"/>'
    b += f'<line x1="95" y1="58" x2="95" y2="196" stroke="#ffffff" stroke-width="2" stroke-dasharray="5 4"/><line x1="88" y1="196" x2="222" y2="196" stroke="#ffffff" stroke-width="2" stroke-dasharray="5 4"/>'
    b += label(104, 135, "深度", TXT, 16)
    b += badge(60, 6, "牙尖（山顶）", WARN) + badge(250, 210, "中央窝（谷底）", GOOD)
    out.append(fig("牙尖和中央窝", "咬合面的“山顶”和“谷底”。位置偏差 = 生成的和真牙的差多远；深度太浅 = 牙面太平。", svg(480, 260, b, "牙尖和中央窝")))
    return [out[0], out[3], out[1], out[2]]  # 两张单图并排，两张“三种情况”各占一整行


# ---------------- 牙齿分割 ----------------
def seg_figs():
    out = []
    # 1. IoU
    g1, g2 = uid("g"), uid("g")
    b = f'<defs>{grad(g1, IVORY)}{grad(g2, CYAN)}</defs>'
    b += f'<ellipse cx="170" cy="130" rx="110" ry="80" fill="url(#{g1})" stroke="{IVORY[2]}" stroke-width="3"/>'
    b += f'<ellipse cx="260" cy="130" rx="110" ry="80" fill="url(#{g2})" fill-opacity=".7" stroke="{CYAN[2]}" stroke-width="3"/>'
    b += label(90, 136, "答案", "#4a3b14", 18, weight="700") + label(300, 136, "预测", "#06343a", 18, weight="700")
    b += label(215, 136, "重叠", "#ffffff", 18, "middle", "800")
    b += label(20, 250, "重叠 ÷ 两块合起来 = 这颗牙的 IoU，所有牙取平均", TXT, 16)
    out.append(fig("T-mIoU", "分出来的每颗牙和标准答案重叠了多少。完全重合 = 100%。分割论文都用它比。", svg(440, 270, b, "IoU")))

    # 2. 牙号全对：牙弓俯视
    b = ""
    import math

    names = ["36", "35", "34", "33", "32", "31", "41", "42", "43", "44", "45", "47"]  # 最后一颗其实是 46，被标成了 47
    teeth = []
    for k, n in enumerate(names):  # 下颌牙弓：U 形（俯视）
        a = math.pi * (k / (len(names) - 1))
        teeth.append((225 - 175 * math.cos(a), 205 - 160 * math.sin(a), n))
    for k, (x, y, n) in enumerate(teeth):
        wrong = k == len(teeth) - 1
        col = (("#ffd1cc", BAD, BAD) if wrong else IVORY)
        g = uid("g")
        b += f'<defs>{grad(g, col)}</defs><ellipse cx="{x}" cy="{y}" rx="22" ry="20" fill="url(#{g})" stroke="{col[2]}" stroke-width="2.5"/>'
        b += label(x, y + 6, n, "#3a2f12" if not wrong else "#5a0d08", 14, "middle", "700")
    b += badge(250, 232, "错了：应该是 46", BAD)
    b += label(225, 150, "下颌（俯视）", DIM, 15, "middle")
    out.append(fig("牙号全对", "一个口扫里每颗牙的牙号都对，才算“全对”。错一颗也不算。", svg(450, 270, b, "牙号")))

    # 3. 幻影牙
    b = gum(10, 190, 420, 60) + molar(30, 90, 110, 110, IVORY) + molar(300, 90, 110, 110, IVORY)
    b += (f'<path d="M170 196 L170 120 Q170 80 220 80 Q270 80 270 120 L270 196 L255 184 L240 196 L225 184 L210 196 L195 184 Z" '
          f'fill="#ffffff" fill-opacity=".18" stroke="{BAD}" stroke-width="3" stroke-dasharray="7 5"/>')
    b += f'<circle cx="203" cy="122" r="7" fill="{BAD}"/><circle cx="237" cy="122" r="7" fill="{BAD}"/>'
    b += badge(150, 26, "幻影牙：这里其实没有牙", BAD)
    out.append(fig("幻影牙", "缺牙的地方只剩牙龈，却被标成了一颗牙。有它就发现不了缺牙。越少越好。", svg(440, 260, b, "幻影牙")))

    # 4. 缺牙检出
    b = gum(10, 190, 420, 60) + molar(30, 90, 110, 110, IVORY) + molar(300, 90, 110, 110, IVORY)
    b += f'<ellipse cx="220" cy="150" rx="62" ry="58" fill="none" stroke="{GOOD}" stroke-width="4" stroke-dasharray="9 6"/>'
    b += badge(160, 26, "找到了：缺 36 ✓", GOOD)
    b += label(220, 160, "?", GOOD, 40, "middle", "800")
    out.append(fig("缺牙检出", "缺了的牙被报出来、牙号也对才算；“放宽”= 报成相邻牙号也算。越多越好。", svg(440, 260, b, "缺牙检出")))
    return out


# ---------------- 颈缘线 ----------------
def prep(x, y, w, h):
    """预备体：磨小了的牙（圆台）。"""
    g = uid("g")
    return (f'<defs>{grad(g, ("#f6ead0", "#d9c69a"))}</defs>'
            f'<path d="M{x},{y + h} L{x + .14 * w},{y + .1 * h} Q{x + .5 * w},{y - .08 * h} {x + .86 * w},{y + .1 * h} L{x + w},{y + h} Z" fill="url(#{g})" stroke="#a8925f" stroke-width="2.5"/>')


def crown_cap(x, y, w, h, bottom_extra, colors=CYAN):
    g = uid("g")
    L, R, B = x, x + w, y + h + bottom_extra
    return (f'<defs>{grad(g, colors)}</defs><path d="M{L},{B} L{L + .1 * w},{y + .3 * h} Q{L + .25 * w},{y - .05 * h} {L + .38 * w},{y + .12 * h} '
            f'Q{L + .5 * w},{y + .25 * h} {L + .62 * w},{y + .12 * h} Q{L + .75 * w},{y - .05 * h} {L + .9 * w},{y + .3 * h} L{R},{B} Z" '
            f'fill="url(#{g})" fill-opacity=".72" stroke="{colors[2]}" stroke-width="2.5"/>')


def margin_figs():
    out = []
    # 为什么重要：三种情况
    def scene(extra, mark_y, ok):
        s = gum(4, 200, 182, 56) + prep(48, 96, 94, 108) + crown_cap(36, 70, 118, 128, extra)
        c = GOOD if ok else BAD
        return s + f'<circle cx="40" cy="{mark_y}" r="11" fill="none" stroke="{c}" stroke-width="3"/>'
    b = panel(0, "短了：有缝", False, scene(-14, 186, False) + label(26, 244, "漏、会蛀", "#5a1d18", 15))
    b += panel(196, "刚好：封住", True, scene(0, 198, True) + label(26, 244, "过渡自然", "#5a1d18", 15))
    b += panel(392, "长了：压牙龈", False, scene(18, 214, False) + label(26, 244, "硌肉、发炎", "#5a1d18", 15))
    out.append(fig("为什么颈缘线要准", "米色：磨好的牙（预备体，种植时是基台）；青：牙冠；粉：牙龈。牙冠边缘要正好对在颈缘线上，误差控制在 0.1–0.2 mm。",
                   svg(582, 262, b, "颈缘线短了、刚好、长了"), wide=True))
    # 颈缘线是什么
    b = gum(10, 186, 420, 70) + prep(150, 66, 140, 130)
    b += f'<path d="M150 194 Q220 214 290 194" fill="none" stroke="{GOOD}" stroke-width="5"/>'
    b += f'<path d="M150 186 Q220 222 290 200" fill="none" stroke="{BAD}" stroke-width="3.5" stroke-dasharray="8 5"/>'
    b += badge(20, 20, "绿：技工画的线", GOOD) + badge(250, 20, "红：模型预测", BAD)
    out.append(fig("颈缘线", "磨好的牙底部、贴着牙龈的那一圈线。预测线（红）和技工画的线（绿）越重合越好。", svg(440, 260, b, "颈缘线")))
    # 平均误差 / 最大误差
    true_c, pred_c = ((30, 160), (220, 60), (410, 160)), ((30, 136), (220, 100), (410, 190))
    b = (f'<path d="M30 160 Q220 60 410 160" fill="none" stroke="{GOOD}" stroke-width="5"/>'
         f'<path d="M30 136 Q220 100 410 190" fill="none" stroke="{BAD}" stroke-width="3.5" stroke-dasharray="8 5"/>')
    for t in (0.12, 0.3, 0.5, 0.7, 0.86):
        (x0, y0), (x1, y1) = qpt(*true_c, t), qpt(*pred_c, t)
        b += f'<line x1="{x0:.0f}" y1="{y0:.0f}" x2="{x1:.0f}" y2="{y1:.0f}" stroke="#ffffff" stroke-width="3" stroke-linecap="round"/>'
    (x0, y0), (x1, y1) = qpt(*true_c, 0.97), qpt(*pred_c, 0.97)
    b += f'<line x1="{x0:.0f}" y1="{y0:.0f}" x2="{x1:.0f}" y2="{y1:.0f}" stroke="{WARN}" stroke-width="6" stroke-linecap="round"/>'
    b += label(20, 230, "白线平均 = 平均误差", TXT, 16) + label(250, 230, "黄线（最长）= 最大误差", WARN, 16)
    out.append(fig("平均误差 / 最大误差", "预测线上每个点到真线的距离：取平均是平均误差，取最大的是 Hausdorff。", svg(440, 250, b, "平均和最大误差")))
    # 0.2 mm 带
    b = (f'<path d="M30 150 Q220 60 410 150" fill="none" stroke="{GOOD}" stroke-opacity=".28" stroke-width="40" stroke-linecap="round"/>'
         f'<path d="M30 150 Q220 60 410 150" fill="none" stroke="{GOOD}" stroke-width="4"/>'
         f'<path d="M30 144 Q130 92 220 108 Q300 120 410 176" fill="none" stroke="{BAD}" stroke-width="3.5" stroke-dasharray="8 5"/>'
         f'<circle cx="395" cy="168" r="12" fill="none" stroke="{BAD}" stroke-width="3"/>')
    b += badge(20, 196, "浅绿带 = 真线两边各 0.2 mm", GOOD)
    out.append(fig("0.2 mm 以内的点", "红线有多少落在浅绿带里。0.2 mm 约等于两个技工画线的差别，进了带就算合格。", svg(440, 240, b, "0.2 毫米带")))
    return out


CSS = """.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:16px}
.fig{background:#0b1426;border:1px solid var(--line);border-radius:14px;padding:14px 16px;display:flex;flex-direction:column;gap:6px}
.fig svg{width:100%;height:auto;display:block}
.fig[style*="grid-column"] svg{max-width:860px;margin:0 auto}
.views{display:flex;flex-direction:column;gap:10px}
.vbar{display:flex;gap:6px;justify-content:flex-end}
.vbtn{background:transparent;color:var(--muted);border:1px solid var(--line);border-radius:16px;padding:4px 14px;font:inherit;font-size:13px;cursor:pointer}
.vbtn.on{background:var(--accent);border-color:var(--accent);color:#04222a;font-weight:700}
.vpane img{width:100%;border-radius:10px;display:block;border:0;background:transparent}
.vpane canvas{width:100%;display:block;border-radius:10px;cursor:pointer}
.v3d{width:100%;min-height:320px;border-radius:10px;overflow:hidden;color:var(--muted);display:flex;align-items:center;justify-content:center}
.v3d canvas{display:block}
.vctl{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;font-size:14px;margin-top:8px}
.vctl input{width:220px;vertical-align:middle}
.vctl output{font-weight:700;min-width:150px}
.vopa{background:transparent;color:var(--fg);border:1px solid var(--line);border-radius:14px;padding:3px 12px;font:inherit;font-size:13px;cursor:pointer}
.vhint{color:var(--muted);font-size:12px}
.fig b{font-family:var(--display);font-size:17px}
.fig span{font-size:14px;color:var(--muted);line-height:1.55}
"""

# 改代号时留下的多余空格（“各取 形状模型 形状误差”）
for name in ("generation.html", "history.html", "ideas.html", "landscape.html", "results.md"):
    p = D / name
    t = p.read_text()
    t2 = re.sub(r"(?<=[一-鿿（“]) 形状模型", "形状模型", t)
    t2 = re.sub(r"形状模型 (?=[一-鿿”）])", "形状模型", t2)
    if t2 != t:
        p.write_text(t2)
        print("spaces", name)

for page, make in (("generation.html", gen_figs), ("segmentation.html", seg_figs), ("margin.html", margin_figs)):
    p = D / page
    s = p.read_text()
    start = s.index('      <div class="inner">\n') + len('      <div class="inner">\n')
    m = re.compile(r'    <div class="figs">\n.*?\n    </div>\n', re.S).search(s, start)
    assert m, page
    new = '    <div class="figs">\n' + "".join(make()) + "    </div>\n"
    s = s[: m.start()] + new + s[m.end():]
    # 样式：去掉旧的 .figs / .fig 规则，换成新的（放大、卡通底色）
    s = re.sub(r"\.figs\{[^}]*\}\n?", "", s)
    s = re.sub(r"\.fig\{[^}]*\}\n?", "", s)
    s = re.sub(r"\.fig svg\{[^}]*\}\.fig b\{[^}]*\}\.fig span\{[^}]*\}\n?", "", s)
    s = re.sub(r"\.fig (svg|b|span)\{[^}]*\}\n?", "", s)
    s = re.sub(r"\.fig\[style\*=\"grid-column\"\] svg\{[^}]*\}\n?", "", s)
    s = re.sub(r"\.(views|vbar|vbtn|vbtn\.on|vpane img|vpane canvas|v3d|v3d canvas|vctl|vctl input|vctl output|vopa|vhint)\{[^}]*\}\n?", "", s)
    s = s.replace("</style>\n</head>", CSS + "</style>\n</head>", 1)
    if "views.js" not in s:
        s = s.replace("</body>", '<script src="views.js" defer></script>\n</body>', 1)
    p.write_text(s)
    print("figs", page)
