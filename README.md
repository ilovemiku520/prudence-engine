<p align="center">
  <img src="docs/assets/overview.svg" alt="睿衡引擎：从客户分布与画像，到价值情景和 AI 辅助解读" width="100%">
</p>

<h1 align="center">睿衡引擎 · Prudence Engine</h1>
<p align="center">面向银行客户的价值分析原型。查看六维画像、比较存贷与理财情景，并在适当性规则和资源约束下生成服务计划。</p>

<p align="center">
  <a href="https://github.com/ilovemiku520/prudence-engine/actions/workflows/tests.yml"><img src="https://github.com/ilovemiku520/prudence-engine/actions/workflows/tests.yml/badge.svg" alt="测试状态"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit UI">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-All_rights_reserved-475569" alt="保留所有权利"></a>
</p>

<p align="center">
  <a href="#quickstart">快速开始</a> ·
  <a href="#features">功能一览</a> ·
  <a href="docs/usage.md">使用与配置</a> ·
  <a href="docs/ai-analysis.md">接入大模型</a> ·
  <a href="docs/customer-scoring-research.md">研究依据</a>
</p>

> **保留所有权利。** 使用本项目受保护的原创内容须事先取得书面许可。完整条款见 [LICENSE](LICENSE)。

默认看板提供 **120 位可复现的模拟客户**。在散点图选择一位客户，查看画像和年度贡献来源，再调整利率比较不同情景。体验统计功能无需配置数据库或模型密钥。[访问演示站](https://prudence-engine-ilovemiku520.streamlit.app/)

这是本科生大数据分析选修课的期末项目。独立的命令行与 REST 决策接口见 [使用与配置](docs/usage.md)。

<a id="features"></a>
## 看板功能

| 你想回答的问题 | 看板中的能力 |
| :--- | :--- |
| 客户如何分布？ | 关注度 × 匹配度、关注度 × 年度贡献、共同因子三个散点视角；筛选、框选、套索与榜单联动 |
| 这个客户有什么特点？ | 点击散点可以查看客户的六维雷达、综合观察分、原始指标与适当性；缺失保持为空 |
| 客户价值从哪里来？ | 存贷款利差、理财费用、预期损失和服务成本拆解；机构贡献与客户净收支分别展示 |
| 换一个经营情景会怎样？ | 自定义利率、FTP、产品收益与费用、资金占比、贷款余额假设，以及产品风险、期限和起投额 |
| 多个指标反映了哪些共同特征？ | 最大似然因子分析 + Varimax；载荷、共同度、独特方差与稳定性诊断 |
| 有限资源优先服务谁？ | 预算、人数与容量约束下的 0–1 分配；只纳入规则可通过且满足配置条件的完整画像 |
| 如何解释这些结果？ | 配置模型接口后，可预览并发送当前群体、所选客户或因子的数值摘要，由模型辅助解读 |

画像使用近期活跃、决策投入、互动响应、风险适配、资金余量和期限匹配六个维度。散点、画像、榜单和服务计划引用同一份客户与产品数据，输入变化会使旧结果失效。

<a id="quickstart"></a>
## 快速开始

需要 Python **3.10+**。持续集成使用 Python 3.11，项目也在 Python 3.12 上完成过本地验证。

```bash
git clone https://github.com/ilovemiku520/prudence-engine.git
cd prudence-engine
python -m venv .venv
```

<details>
<summary>激活虚拟环境：选择你的系统</summary>

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

</details>

```bash
python -m pip install -r requirements.txt
python -m streamlit run ui.py
```

在浏览器打开 [本地看板](http://localhost:8501)。看板直接启动模拟客户分析。独立的 CLI 与 REST 决策接口另需安装 `requirements-engine.txt`，其合成数据意图模型属于原型演示。

1. 在散点中选择一位客户，查看右侧雷达与贡献拆解。
2. 打开左侧“利率与理财策略”，修改参数并点击“应用利率与策略”。
3. 在“共同因子与稳定性”核对指标结构，或在“服务计划”设置资源约束。
4. 在“AI 辅助解读”填入自己的接口配置，预览摘要后手动发送。

需要导入客户数据、连接数据库或启动 REST 服务？查看 [使用与配置](docs/usage.md)。`.env.example` 是环境变量清单，**复制成 `.env` 不会自动加载**。

## 多模型 AI 辅助解读

| 接口家族 | 可配置的服务 |
| :--- | :--- |
| Responses / Chat Completions | OpenAI；Azure OpenAI v1 按兼容方式配置 |
| OpenAI 兼容 Chat | DeepSeek、通义千问 / 百炼、Ollama 及其他兼容网关 |
| Messages | Claude / Anthropic |
| Generate Content | Gemini |

模型名、地址、密钥和输出上限均由用户配置。只有点击发送才调用外部服务；默认发送去除姓名与客户 ID 的数值摘要，页面密钥只保留在当前会话。去标识摘要仍可能包含敏感业务信息，发送前请核对预览。

自动测试以模拟响应覆盖接口协议。关于本地模型、Azure v1、鉴权和解释 API，见 [AI 接入指南](docs/ai-analysis.md)。

## 计算链路

```mermaid
flowchart LR
    D[模拟 / 导入 / 配置数据源] --> V[字段校验与客户特征]
    V --> R[适当性硬规则]
    V --> S[六维观察评分]
    V --> F[共同因子分析]
    P[利率与产品情景] --> C[年度贡献拆解]
    V --> C
    R --> B[统一散点与客户画像]
    S --> B
    F --> B
    C --> B
    B --> O[受约束的服务计划]
    B --> A[预览摘要 → 手动 AI 解读]
```

看板的观察评分与独立决策接口的 XGBoost 意图分各有定义。后者通过 `Prudence → Intent → Aegis` 完成适当性优先的融合决策；AI 只解释已有结果，不重写任何评分或规则。

## 文档导航

| 文档 | 内容 |
| :--- | :--- |
| [使用与配置](docs/usage.md) | 页面操作、数据字段、环境变量、CLI 与 API |
| [计算方法](docs/analytics-methods.md) | 评分公式、价值口径、因子分析与优化约束 |
| [研究到实现](docs/customer-scoring-research.md) | 论文、开源参考及实现取舍 |
| [AI 接入](docs/ai-analysis.md) | 模型协议、密钥、发送摘要和错误处理 |
| [维护与迁移](docs/maintenance.md) | 验证方式、代码分层、旧入口迁移与 README 设计参考 |
| [部署与性能](docs/deployment.md) | 启动优化、免费平台限制、Docker 和 Render 备用方案 |

## 开发与验证

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

单元与回归测试覆盖评分、适当性、金额守恒、因子结构、服务分配、数据导入、页面联动及 AI 协议。持续集成还会构建看板容器，并检查页面加载和因子散点切换。

```text
ui.py → dashboard.py           统一看板入口与客户交互
customer_scoring.py            六维评分、价值情景与因子分析
dashboard_data.py              视觉样式、导入与导出
dashboard_ai.py                AI 配置、摘要预览与结果展示
analysis_context.py            数值白名单与结果失效判定
ai_explainer.py                多模型文本接口适配
analytics.py                  分配优化与 PCA / SVD / 博弈数值工具
main.py → api.py               命令行、引擎组装与 REST 接口
data_source.py                模拟、内存与关系数据库适配
workbench_data.py             导入校验与可复现模拟样本
tests/ · docs/                回归测试与方法文档
```

## 结果应该怎样理解

- **观察分 ≠ 购买概率或信用评级**；高分不能覆盖 `RESTRICTED / FORBID`。
- **年度贡献 ≠ 实际利润或客户终身价值**；采用固定余额的一年简化情景，利率不是实时报价。
- 因子与相关性用于探索，不能证明因果；AI 文本须对照计算值复核。
- 项目是分析与决策原型。生产身份认证、限流、高可用及持久化审计需由部署环境实现；当前进程内审计与指标会在重启后丢失。

---

<!-- BEGIN RIGHTS NOTICE -->
## 版权与使用限制 / Copyright and use restrictions

**保留所有权利。未经著作权人事先书面许可，不得使用、运行、复制、修改或分发本项目受保护的原创内容，包括个人、学习、研究、非商业和商业用途，以及依法需要许可的 AI 使用。**

**All rights reserved. Prior written permission is required to use, run, copy, modify or distribute the project's protected original material, including personal, educational, research, non-commercial and commercial use, and AI use where permission is required by law.**

完整条款见 [LICENSE](LICENSE)。第三方内容仍适用其各自许可；此前已授予的许可、法定权利及 GitHub 平台条款项下权利不受影响。本文中的安装、运行及开发说明仅为技术说明，不构成使用授权。

See [LICENSE](LICENSE) for the full terms. Third-party licenses, previously granted permissions, statutory rights and rights under GitHub's Terms of Service remain unaffected. Setup, usage and development instructions are technical documentation, not permission to use the material.

书面授权 / Permission requests: [ilovemiku520@outlook.com](mailto:ilovemiku520@outlook.com)

关注初音未来谢谢喵，ilovemiku520  
Please follow Hatsune Miku, thank you, meow. ilovemiku520
<!-- END RIGHTS NOTICE -->

作者：[@ilovemiku520](https://github.com/ilovemiku520) · [联系作者](mailto:ilovemiku520@outlook.com)
