# 从研究到统一客户看板

检索与实现日期：2026-09-29。所有默认客户均为模拟数据；以下来源是设计与方法参考，不构成本项目效果已被验证的证据。

| 一手来源 | 学习内容 | 本项目落实 | 没有声称的能力 |
|---|---|---|---|
| [llm2014/llm_benchmark](https://github.com/llm2014/llm_benchmark)，考察提交 `59e1e9f0b724d655da271fbc32afc7f0aeae0c61` | 同一筛选状态连接散点和榜单；明确轴含义，支持指标切换和分区阅读 | 客户作为点，产品与人群筛选贯穿散点、榜单、雷达和服务计划 | 参考项目评测的是模型；其测试分不能直接迁移为客户分。未复制其代码或素材 |
| [Shneiderman (1996), The Eyes Have It](https://hci.stanford.edu/courses/cs448b/papers/shneiderman96eyes.pdf), DOI: 10.1109/VL.1996.545307 | 先总览，再缩放/筛选，按需查看明细 | 散点总览 → 点选客户 → 同页雷达、原始指标和贡献拆解；保留客户选择器作为替代操作 | 未声称雷达图本身具有预测能力 |
| [Aliyev et al. (2020), Segmenting Bank Customers via RFM Model and Unsupervised Machine Learning](https://arxiv.org/abs/2008.08662) | 用交易行为和无监督方法探索银行客户群体 | 借鉴行为分层的问题意识；购买间隔、浏览及互动参与画像 | 无完整交易频次和交易金额流水，因此不是标准 RFM；无利润/留存时间序列，因此不是 CLV |
| [OECD / JRC (2008), Handbook on Constructing Composite Indicators](https://doi.org/10.1787/9789264043466-en) | 复合指标需要解释变量选择、归一化、权重、聚合与敏感性 | 固定锚点、缺失不补零、权重可改、几何平均及排名敏感性图 | 本项目的 20 次浏览、30 天半衰期等锚点为工程约定，不是该手册推荐的银行参数 |
| [scikit-learn 因子分析文档](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html)与[旋转示例](https://scikit-learn.org/stable/auto_examples/decomposition/plot_varimax_fa.html) | 潜在共同因子与变量独特噪声分开；旋转辅助结构解释 | 标准化的最大似然因子分析、Varimax、载荷/共同度/独特方差、稳定符号方向与诊断 | 因子不是 PCA 改名，也不是客户价值分；旋转不保证预测效果变好 |
| [Grant (2011), Liquidity transfer pricing: a guide to better practice, BIS / FSI Occasional Paper 10](https://www.bis.org/fsi/fsipapers10.pdf) | 资金占用有成本，稳定资金来源有内部收益；期限和流动性条件不可忽略 | 以可编辑 FTP 展示存款与贷款利差的简化机构口径 | 单一 FTP 是教学简化，未实现机构级期限曲线、流动性溢价或监管资本模型 |

## 实际学习过程

阅读参考仓库的 README、`docs/assets/charts.js` 中散点渲染、`docs/assets/benchmark-domain.js` 中指标口径，以及页面筛选与排行榜结构。将“同一对象集贯穿图表”的交互思想迁移到本项目，独立编写 `customer_scoring.py` 和统一页面。

阅读银行分群论文摘要与全文、复合指标手册的归一化/权重/敏感性部分、交互可视化论文，以及 scikit-learn 的因子分析实现说明。研究引用直接显示在看板“口径与研究”中；精确实现约定集中在 [analytics-methods.md](analytics-methods.md)。

## 为什么分成三个视角，但仍是同一看板

观察分衡量当前可见信号；年度贡献衡量特定金额和利率假设下的机构贡献；因子揭示变量共同变化。它们的单位和用途不同，不应强行加总为一个看似精确的“客户好坏分”。三个散点视角通过同一客户 ID、雷达与筛选关联，客户无需在无关工具页间跳转。

## 后续真实验证需要什么

若要升级为实证客户价值模型，需要经授权的历史余额、实际利息/费用/成本、交易时间、触达记录、留存与转化标签。届时应按时间切分训练/验证数据，比较简单基线，校准收益与风险假设，评估跨群体稳定性。当前合成样本上的图形和数值检查只能验证实现，不证明业务有效性。
