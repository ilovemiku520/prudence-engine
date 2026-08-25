# 睿衡引擎（Prudence Engine）

银行理财导购场景的适当性与意图联合决策原型。它演示如何把不可逾越的合规规则放在最高优先级，再结合客户行为意图，输出拦截、人工复核、培育或促单建议。

[在线演示](https://prudence-engine-ilovemiku520.streamlit.app/) · 仅使用模拟或用户主动导入的数据

> 这是教学与作品集项目，不是经过银行生产验证的系统，也不能代替持牌机构的合规审查、风险测评或人工决策。

## 解决的问题

普通推荐模型只追求点击或转化，可能把高风险产品推荐给承受能力不足的客户。本项目把决策拆成三层：

1. `Prudence` 根据 C1–C5 客户等级、R1–R5 产品等级、高龄、首次购买、资产与期限规则给出 `ALLOW / RESTRICTED / FORBID`；
2. `Intent` 用规则分和 XGBoost 模型估计意图，并用 SHAP 给出主要影响信号；
3. `Aegis` 以适当性结果为硬约束融合意图分，生成可解释行动建议。

未知客户或产品会明确失败，不再用默认画像替代真实输入，避免产生“看似成功但依据错误”的决策。

## 当前实现边界

| 能力 | 状态 | 说明 |
|---|---|---|
| 适当性矩阵与扩展规则 | 已实现 | 确定性 Python 规则，可直接测试 |
| 规则分 + XGBoost 意图融合 | 已实现原型 | 缺少真实业务标签时自动使用合成数据训练 |
| SHAP 解释 | 已实现原型 | 解释当前模型输出，不等于因果解释 |
| Streamlit 单人/多人分析 | 已实现 | 支持 CSV、Excel、JSON 导入与结果导出 |
| FastAPI 决策接口 | 已实现 | 输入约束、批量上限、CORS 白名单与受保护管理接口 |
| SQLite/MySQL/PostgreSQL | 已实现适配层 | 需要用户自行提供数据库和表结构 |
| Redis | 可选适配 | 默认关闭；不可用时回退进程内缓存 |
| Hive/Spark/Kafka/Flink | 未实现 | 仅是未来生产化方向，不属于当前仓库能力 |
| 生产级鉴权、限流、高可用 | 未实现 | 上线前必须由部署环境补齐 |

仓库不再展示没有真实数据、实验脚本和置信区间支撑的 AUC、准确率或转化提升数字。

## 快速开始

要求 Python 3.10+。

```bash
git clone https://github.com/ilovemiku520/prudence-engine.git
cd prudence-engine
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env   # Windows；macOS/Linux 使用 cp
streamlit run ui.py
```

命令行单次决策：

```bash
python main.py --mode decision --customer CUST_HIGH --product P004
```

启动 API：

```bash
python api.py
```

API 文档默认位于 `http://127.0.0.1:8000/api/docs`。

## 安全默认值

- `.env`、Streamlit secrets、日志、数据库和模型文件不会进入版本控制；
- 审计日志保存客户和产品标识的盐化哈希，不记录原始 ID；
- `/api/customers`、`/api/customer/*`、`/api/metrics`、`/api/audit` 需要 `X-Admin-Token`；
- 未设置 `PRUDENCE_ADMIN_TOKEN` 时，管理接口保持关闭；
- 批量决策每次最多 100 条，标识只允许有限字符和长度；
- CORS 默认只允许本地 Streamlit 地址。

公开部署前请生成高强度 `PRUDENCE_ADMIN_TOKEN` 与 `PRUDENCE_AUDIT_SALT`，并在网关层补充 TLS、身份认证、速率限制、密钥托管和持久化审计。

## 数据格式

客户：

```csv
id,risk,age,assets,period,first_buy,name,income
CUST_001,C3,35,800000,365,false,示例客户,中
```

产品：

```csv
id,risk,name,lock,min,type
P001,R1,示例货币产品,0,0,货币型
```

意图特征：

```csv
customer_id,feature_name,feature_value
CUST_001,beh_calculator_use_cnt,3
CUST_001,beh_view_cnt_7d,8
```

## 验证方法

```bash
pip install -r requirements-dev.txt
pytest -q
```

真实效果评估必须使用按时间切分的业务数据，至少报告样本量、类别比例、AUC/PR-AUC、校准误差、适当性规则用例覆盖率和分组置信区间。合成数据报告只能验证代码链路，不能证明业务提升。

## 项目结构

```text
prudence_suitability.py  适当性矩阵与扩展规则
intent_subsystem.py      规则分、XGBoost 与 SHAP
aegis_decision.py        适当性优先的融合决策
nexus_orchestrator.py    数据、特征与引擎编排
data_source.py           模拟/文件/关系数据库适配
api.py                   FastAPI 服务与管理接口保护
ui.py                    Streamlit 交互仪表板
validity_evaluator.py    合成数据链路评估工具
tests/                   安全边界与领域规则回归测试
```

## 已知限制

- 默认模型由合成标签训练，只用于展示流程；
- 进程内指标和审计在重启后丢失，不适合生产追溯；
- 上传数据的隐私、授权和保存期限由部署者负责；
- 规则矩阵需要结合具体机构制度、监管要求和法务意见重新确认。

## 许可证与作者

代码采用 [MIT License](./LICENSE)。

作者：[@ilovemiku520](https://github.com/ilovemiku520) · ilovemiku520@outlook.com
