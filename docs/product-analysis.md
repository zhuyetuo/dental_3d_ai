# 竞品分析与产品功能规划

目标定位：**智能化、轻量化的 Web 版牙科 CAD**。对标 dentalai.ai 的 AI 自动设计，同时补上 3Shape / exocad 这类桌面软件的交互编辑能力，全部在浏览器里完成。

## 1. dentalai.ai 解析

来源：用户保存的两个页面——官网首页（dentalai.ai），以及登录后的批处理上传页（app.dentalai.ai）。在线页面在当前环境里无法直接访问。

### 1.1 技术栈（从页面源码判断）

| 项 | 观察 |
|---|---|
| 前端框架 | Next.js（App Router，页面里有 `_next/`、RSC `self.__next_f`） |
| UI 组件 | 应用端用 MUI（Material UI），官网用 Tailwind |
| 3D | 官网加载了 three.js r180 |
| 支付 | 引用了 Stripe |
| 域名 | 官网 dentalai.ai，应用 app.dentalai.ai |

> 页面里的 Element Plus 弹窗（“温馨提示”）不属于该网站，应该是浏览器插件注入的。

### 1.2 官网宣传的核心流程（3 步）

首页轮播图展示三步：

1. **Auto Segmentation（自动分割）**：自动识别并分割单颗牙齿和整个牙弓，生成干净、结构化的数据，供后续步骤使用（截图 `crown_mandible_top.jpg`）；
2. **Auto Margin Detection（自动颈缘线检测）**：截图 `margin_mandible_top.jpg`；
3. **AI Crowns（AI 牙冠）**：“即时设计完全贴合的单冠和多冠修复体，自带自然解剖形态”。

卖点：从扫描到牙冠一条流程、颈缘线和牙冠精度稳定、批量自动化处理。
模式是 **AI 加人工专家复核**（用户评价里提到 hybrid AI and expert review）。

FAQ 透露了重点能力：咬合和邻接触点控制、兼容现有 CAD/CAM、前牙和全牙弓修复、患者数据安全、模型持续学习。

### 1.3 应用端（app.dentalai.ai）

**导航**：Batch Dashboard、Batch Process、Upload、Dashboard、Subscription、Settings、Credit History、Account、Feedback、Help center。
商业模式是**订阅加点数（credit）计费**。

**批量上传**：
- 先选修复体类别，再配置参数，最后拖入文件夹；
- 每个病例一个子文件夹，**里面正好 2 个 STL（上颌 + 下颌）**；
- 每批最多 100 个病例。

**修复体类别**：Crown、Nightguard / Bite-Splints（咬合垫）、Bridge、Model Builder（模型制作）、Inlay / Onlay（嵌体 / 高嵌体）、Coping（内冠）、Custom Tray（个别托盘）、Miscellaneous。

**Crown 设计参数（默认值）**：

| 参数 | 默认值 | 含义 |
|---|---|---|
| Contacts Distance | -0.03 mm | 与邻牙的距离，负值表示轻微过接触 |
| Occlusion Distance | 0.30 mm | 与对颌的距离 |
| Undercuts | Remove | 去除预备体倒凹 |
| Emergence Profile | Regular | 穿龈轮廓 |
| Contacts Shape | Regular | 邻接触区形状 |
| Style | Facial Cutback | 唇面回切，用于上瓷 |
| Cement Gap | 0.03 mm | 粘接间隙 |
| Crown Thickness | 0.50 mm | 最小冠厚度 |

这些参数已经放进本项目的 `ml/dental_ai/preferences.py`，前端和 API 都可以传入。

### 1.4 结论

- dentalai.ai 的主线是**预备体牙冠**：分割 → 颈缘线 → 牙冠（冠内表面贴合预备体）。这正是我们文档里的“场景 B”，**需要提前到 M2/M3**；
- 它的形态是**批处理服务**：上传、排队、下载结果，没有在线 3D 编辑。在线编辑正是我们可以做得更好的地方；
- 输入固定为“上下颌两个 STL”，因为对颌是咬合设计的必需输入。

## 2. 3Shape 与先临三维

**3Shape**
- Dental System 是桌面 CAD 的行业标准，交互编辑能力最全；
- 3Shape Automate 是云端 AI 设计服务：上传扫描，自动分析和测量，设计牙冠解剖形态，冠内表面仍由 Dental System 生成。2024 年起扩展到前牙；
- 厂商宣称 94% 的设计接受率，这是厂商自己的数字。2026 年的第三方对比研究中，Automate 的偏差比 Dentbird 大，多数指标也不如 exocad AI Design，说明这个领域还有追赶空间。

**先临三维（Shining 3D）**
- 口扫仪端（Aoralscan 3）用 AI 实时识别并过滤多余软组织；
- 在 IEEE DataPort 上公开过一个[牙齿分割数据集](https://ieee-dataport.org/documents/shining3d-tooth-segmentation-dataset)，可以作为 Teeth3DS 之外的训练和评估数据（需核对许可）；
- 公开资料里没有找到其设计软件 AI 分割的技术细节。用户反馈它的分割效果很好，可以拿它的分割结果作为我们的质量标杆，建议拿同一批扫描做对比测试。

## 3. 我们的功能规划

在 dentalai.ai 功能的基础上，按优先级排列：

| 优先级 | 功能 | dentalai.ai | 我们 |
|---|---|---|---|
| P0 | 上下颌双 STL 输入、病例管理 | ✅ | 已支持单颌加对颌，改成成对上传 |
| P0 | 自动分割 + FDI 编号 | ✅ | 接口和后处理已完成，待接模型 |
| P0 | **在线 3D 查看和编辑**（改牙号、画笔修正分割） | ❌ 批处理 | 查看已完成，编辑待做 —— **核心差异化** |
| P0 | 缺失牙生成（种植冠 / 桥体），底面贴合牙龈 | 未提及 | V1 已完成 |
| P1 | 自动颈缘线检测 + 在线编辑 | ✅ 检测 | 已完成（几何方法 + 编辑器） |
| P1 | 预备体牙冠：粘接间隙、最小厚度、去倒凹 | ✅ | 已完成（含颈缘精确切割、邻接调整） |
| P1 | 设计参数（邻接、咬合、穿龈、风格） | ✅ | 邻接距离、咬合距离、粘接间隙、冠厚度、去倒凹已生效；穿龈轮廓、接触区形状、风格待做 |
| P1 | 批量处理（文件夹、最多 100 例）、批次看板 | ✅ | 待做，后端任务队列已预留 |
| P2 | 桥（多单位、连接体截面积约束） | ✅ | 待做 |
| P2 | 嵌体 / 高嵌体、内冠、唇面回切 | ✅ | 待做 |
| P2 | 咬合垫、个别托盘、模型底座（Model Builder） | ✅ | 待做，主要是几何算法，不太依赖 AI |
| P2 | 订阅、点数计费、支付 | ✅ Stripe | 国内对接微信 / 支付宝 |
| P3 | 咬合接触热力图、邻接距离热力图 | 未提及 | 差异化 |
| P3 | 医生修正数据回流训练 | “持续学习” | 差异化 |
| P3 | 正畸：牙齿移动、排牙 | ❌ | 分割做好后的自然延伸 |

## 4. 对技术路线的影响

1. **场景 B（预备体牙冠）提前**：颈缘线检测和冠内表面生成从 M4 提前到 M2/M3，因为这是行业里最主流的单（dentalai.ai、3Shape Automate 的主线都是它）；
2. **成对上传**：API 改为一次上传上下颌两个 STL，自动判断哪个是上颌（可以用分割模型的置信度判断，也可以用文件名）；
3. **批处理**：引入 Celery + Redis，按批次统计进度；结果打包下载（ZIP：每颗牙一个 STL，加一份报告）；
4. **前端编辑器**：这是相对 dentalai.ai 最大的差异化，投入优先级要和 AI 模型同级。
