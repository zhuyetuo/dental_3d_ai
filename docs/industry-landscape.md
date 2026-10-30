# 行业格局：产业界、学术界与 AI 的实际提效

> 调研时间：2026 年 10 月。厂商数字大多是厂商自己宣称的，文中会标明；独立研究的数据单独列出。

## 1. 这属于什么领域

| 视角 | 叫法 |
|---|---|
| 产业 / 产品 | **AI 牙科 CAD**、**智能修复设计**（AI dental CAD / AI-driven restoration design），属于**口腔数字化（Digital Dentistry）**中的 **CAD/CAM** 环节 |
| 临床学科 | 口腔修复学（Prosthodontics）中的数字化修复（Digital Prosthodontics），服务对象主要是**义齿加工厂（技工所）**和椅旁（诊所内）修复 |
| 学术 / 技术 | 医学图像计算（MICCAI 社区）+ 3D 几何深度学习（点云 / 网格分割、形状补全、3D 生成模型）+ 计算机图形学的网格处理 |

“智能牙科 CAD” 这个叫法是准确的。更细地说，我们做的是 **“Web 端 AI 牙科 CAD”**：
dentalai.ai、Dentbird 是同类产品；3Shape Automate 是桌面 CAD 巨头推出的云端 AI 服务；先临三维的 AI Design 是扫描仪厂商补齐的设计能力。

### 数字化修复流程，以及 AI 介入的环节

```
口扫 / 石膏扫描 → 建模与分割 → 颈缘线 → 牙冠设计 → 细节调整与复核 → CAM（切削 / 3D 打印）→ 试戴
                    AI ✔         AI ✔       AI ✔        人工为主         —
```

传统做法：技工在 3Shape / exocad 里手动完成分割、画颈缘线、从牙库选牙、调形态、调邻接和咬合，也就是“手撸”。
AI 做法：前三步自动完成，技工主要做**复核和微调**。

## 2. 产业界

### 2.1 国际

| 公司 / 产品 | 形态 | 已公开的信息 |
|---|---|---|
| **3Shape Automate** | 桌面 CAD 巨头的云端 AI 设计服务 | 单冠 2.99 美元、约 90 秒交付；能等 10 小时则 1.99 美元；宣称已下载超过 500 万个设计病例；2024 年扩展到前牙；宣称 94% 设计接受率（均为厂商数据） |
| **exocad**（AI Design） | 桌面 CAD 内置 AI 方案 | 与 3Shape 并列市场最大的两家 CAD 软件公司；2026 年一项对比研究中，多数指标优于 3Shape Automate |
| **Dentbird**（Imagoworks，韩国） | **浏览器端** AI 牙冠设计 | 与我们形态最接近；一份单病例报告中，约 80% 形态直接沿用 AI 结果，约 20% 做了微调 |
| **dentalai.ai** | Web 批处理 AI 设计 | 详见 [product-analysis.md](product-analysis.md) |
| **Glidewell**（CrownAI） | 美国最大技工所之一，自研 AI | 用在自己的加工流程和椅旁系统里 |
| **relu** | AI 牙科自动化（CBCT 分割等） | 种子轮融资 1800 万（美元） |
| 其它 | Formlabs 兼容的 AI 设计服务 | Dentscape、CADflow、H3D、SimpliOne、Ai4Dental、DIBS AI 等，说明赛道已经很拥挤 |

### 2.2 国内

| 公司 | 相关能力 | 已公开的信息 |
|---|---|---|
| **先临三维（Shining 3D）** | 2025 年 9 月推出 **AI Design**，是齿科云平台的核心模块，做牙冠和模型的智能设计，并与扫描、3D 打印打通 | 2026 北京口腔展上宣传单冠生成约 60 秒；齿科产品累计覆盖全球 10 万余家诊所和加工厂（厂商数据） |
| **菲森科技** | 全流程数字化方案 + AI 修复冠设计 | 宣传单冠平均 1.5 分钟（厂商数据） |
| **美亚光电** | 以 CBCT 影像为主，美亚智云平台 | AI 设计细节公开较少 |
| **时代天使** | 隐形矫治 | 与牙冠设计关系较弱，但牙齿分割是其正畸方案设计的基础能力 |

### 2.3 市场信号

- **技工短缺**：多份行业报告把“合格技工短缺”列为技工所产能的主要瓶颈；美国劳工统计局也提到正规技工培训项目在减少。**这是 AI 设计最直接的需求来源**；
- 市场规模：各家机构估计差距很大，只能看量级。全球技工所市场约 270 亿–390 亿美元（2026）；牙科 CAD/CAM 约 48 亿美元（2026，年增 9.6%）；牙科 AI 约 5.8 亿美元（2026，预计 2033 年 19.6 亿）；
- 价格锚点：3Shape Automate 单冠 2–3 美元。**AI 设计的单价会被压得很低，商业模式要靠量（批处理、订阅）和差异化能力（在线编辑、更多修复类型、本地化）**。

## 3. 学术界

| 方向 | 代表工作 | 现状 |
|---|---|---|
| 牙齿分割与编号 | TGNet（3DTeethSeg'22 第一名）、3DTeethSAM（AAAI 2026）、TSegAgent（MICCAI 2026） | 公开集上已经饱和（mIoU 92% 以上）；**难点在跨来源泛化**，换一种扫描来源，3D 专用网络 mIoU 会跌到约 40% |
| 关键点检测 | 3DTeethLand'24 挑战赛、ToothLDNet（MICCAI 2025） | 首个公开的 3D 牙齿关键点数据集来自 3DTeethLand |
| 颈缘线检测 | Alsheghri 等（2024–2026）、MADCrowner 的 CrownSegger | 数据集小、多为单中心 |
| 牙冠生成 | DMC（MICCAI 2023）、DCrownFormer+（MedIA 2025）、VBCD（MICCAI 2025）、MADCrowner（MedIA 2026）、ToothCraft（VISAPP 2026）、CrownGen | 技术从点云补全演进到模板变形和扩散模型；**咬合面细节（窝沟、牙尖）仍是难点** |
| 临床评价 | 多篇 2026 系统综述和荟萃分析；在研临床试验 NCT07834957、NCT07562607 | 结论偏谨慎，见下一节 |

**主要研究力量**：
- 加拿大蒙特利尔理工学院（DMC、颈缘线系列）；
- 浙江大学（周昆、郑友怡团队：3DTeethSAM、ToothLDNet）；
- 香港中文大学 MMLab（VBCD、MADCrowner）；
- 首尔大学（DCrownFormer）；
- 山东大学 / 萨里大学 / 伦敦国王学院（TSegAgent）；
- 突尼斯斯法克斯 CRNS（Teeth3DS 数据集和挑战赛）；
- 德国雷根斯堡大学（Broll、Rosentritt：AI 牙冠临床与材料评价）。

**学术界和产业界的差距**：论文大多只做“牙冠外形”，很少有人做完整的可加工输出，包括封闭网格、粘接间隙、最小厚度、去倒凹、邻接和咬合参数化。这些恰恰是产品必须做的工程部分，也是我们的骨架里已经开始做的（`margin/base_surface.py`、`preferences.py`）。

## 4. AI 能减少多少人工、加快多少

### 4.1 研究数据

| 来源 | 结论 |
|---|---|
| 系统综述（PMC，2026） | 设计时间减少约 **30–50%**；但很多研究没有把“设计时间”和“后续修整时间”分开统计 |
| GAN 牙冠综述 | 比技工 CAD 流程减少 **60% 以上**，形态相当或更好 |
| Dentbird 与 3Shape CAD 对比（实验室研究） | 单颗上颌冠约 99 分钟对 397 分钟，减少约 **75%**（统计显著） |
| 另一项牙支持式冠研究 | 4.5 分钟对 6.2 分钟（熟练专家），只减少约 27% |
| Journal of Dentistry 体外研究（2026） | 每颗冠节省 9–18 分钟（因牙位而异）；但**边缘密合度和咬合接触质量两种流程都有问题**，AI 没有改善关键临床指标 |
| BMC Oral Health 荟萃分析（2026） | AI 显著减少设计时间，咬合形态相当 |
| Archives of Health Investigation 综述（2026） | 合并后设计时间、咬合形态、内部密合**都没有显著差异**，研究之间异质性很高（I² > 96%） |
| PMC 研究 | AI 设计的牙冠**一致性好，与技工经验无关**，新手和老手结果相近 |
| 3Shape（厂商） | 后牙单冠从 15 分钟降到 90 秒 |

### 4.2 综合判断

以下是根据上面的数据做的估计，不是测量结果。

1. **单冠设计时间**：熟练技工手动设计一颗后牙单冠大约 10–20 分钟，复杂病例更久。AI 生成只要 1–2 分钟，技工复核和微调 3–8 分钟。**单颗设计时间减少约 50–70%，单个技工的产能大约提高 2–3 倍**；
2. **人力**：AI 目前**不能取代技工**，所有临床研究都要求人工复核咬合、邻接和美观，复杂病例仍需专家修整。可以替代的是“初级技工从头画”的那部分工作。对技工所来说，**同样的产量大约可以少用 30–50% 的设计人力**，或者人数不变、接更多单；
3. **最省时的环节**：分割和颈缘线（手动最繁琐，AI 能从几分钟降到几秒）、后牙单冠（最标准化）；
4. **最难省的环节**：前牙美学、多单位桥、复杂咬合、种植上部结构。人工占比仍然高；
5. **质量**：AI 带来的是**一致性**（不依赖技工水平），而不是绝对更好。边缘密合和咬合接触仍然要人工把关。

### 4.3 对我们产品的启示

- 宣传口径不要说“替代技工”，要说“**技工产能翻倍、新手也能出稳定的设计**”；
- **复核和微调的体验决定实际节省多少时间**：AI 生成只占 1–2 分钟，剩下的时间都在编辑器里。在线编辑器（改牙号、拖颈缘线、调邻接和咬合并实时显示距离热力图）的投入要和 AI 模型同级；
- 要做自己的效率评估：和合作技工所做 A/B 对比，统计每个病例的**总时间（含修改）**、修改次数、返工率、试戴调磨量，而不只是看 AI 生成时间。

## 参考来源

- [Head-to-Head: AI and Human Workflows for Single-Unit Crown Design—Systematic Review (Oral, 2026)](https://doi.org/10.3390/oral6010016)
- [Accuracy and Functional Performance of AI-Based Automated Crown Design Systems: Systematic Review and Meta-Analysis (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13305636/)
- [In vitro evaluation of automated single-unit crown design using AI (Journal of Dentistry, 2026)](https://pubmed.ncbi.nlm.nih.gov/42142614/)
- [AI in dental restoration for designing crowns: a meta-analysis (BMC Oral Health)](https://pubmed.ncbi.nlm.nih.gov/42271364/)
- [AI Designing Tooth-Supported Fixed Prosthesis: a Systematic Review](https://archhealthinvestigation.emnuvens.com.br/ARCHI/article/view/6754)
- [AI-based CAD system design time and uniformity (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11886404)
- [Dentbird Crown single posterior case (Dental Tribune)](https://www.dental-tribune.com/news/layout-4-pages-ai-based-chairside-crown-design-a-single-posterior-case-using-dentbird-crown/)
- [Clinical trial NCT07834957](https://clinicaltrials.gov/study/NCT07834957)、[NCT07562607](https://clinicaltrials.gov/study/NCT07562607)
- [Top Dental AI Companies 2026](https://www.ddschoology.com/top-dental-ai-companies/)
- [Formlabs dental design integrations](https://dental.formlabs.com/software/dental-design-solutions-and-integrations/)
- [Best AI Platforms for Digital Dentistry（含 3Shape Automate 价格）](https://dentistdecoded.com/articles/best-ai-platforms-for-digital-dentistry/)
- [Digital data acquisition and CAD technology in Dentistry (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2588914125000115)
- [先临三维 AI 设计亮相北京口腔展（雪球）](https://xueqiu.com/7214020925/394653792)
- [2026 口腔数字化管理方案（搜狐）](https://m.sohu.com/a/1023147717_120596020/)
- [中国 9 大牙科上市企业中报（界面新闻）](https://www.jiemian.com/article/15062592.html)
- [2025 中国口腔医疗 AI（网易）](https://www.163.com/dy/article/KIM8B8G60538AB6H.html)
- [relu 1800 万种子轮](https://www.trysignalbase.com/news/funding/relu-raises-18m-seed-round-for-ai-dental-automation)
- 市场规模：[Towards Healthcare](https://www.towardshealthcare.com/insights/dental-lab-market-sizing)、[MarkWide Research](https://markwideresearch.com/dental-cadcam-market)、[Persistence Market Research](https://www.persistencemarketresearch.com/market-research/ai-in-dentistry-market.asp)
- [3DTeethLand challenge (arXiv 2512.08323)](https://arxiv.org/html/2512.08323v1)
