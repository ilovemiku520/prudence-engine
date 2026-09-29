# 维护与迁移

[返回 README](../README.md) · [使用与配置](usage.md)

## 验证当前实现

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
```

测试包含可复算的领域约束和页面交互。需要指定临时目录时可使用 `--basetemp=path/to/test-temp`，不要将已有资料目录用作测试临时目录。

真实业务效果评估仍需带时间与结果标签的数据、独立验证集、样本量、类别比例、AUC / PR-AUC、校准误差和置信区间。合成样本、规则测试通过或模型解释图都不能证明转化提升。

## 代码职责

- 看板：`dashboard.py` 组织交互，`dashboard_data.py` 处理样式和导入，`dashboard_ai.py` 处理 AI 页面。
- 分析：`customer_scoring.py` 负责观察评分、年度贡献和因子分析；`analytics.py` 保留整数规划、PCA、SVD 和博弈数值函数。后者仍有测试和研究用途，不作为独立学科页面展示。
- AI：`analysis_context.py` 提供白名单摘要，`ai_explainer.py` 负责协议与网络限制；不能由模型输出修改分数或适当性。
- 决策：`main.py` 装配引擎，`nexus_orchestrator.py` 协调特征，`prudence_suitability.py`、`intent_subsystem.py`、`aegis_decision.py` 保留适当性优先的决策链路；`api.py` 提供接口。

## 2026-09：旧入口清理

| 移除内容 | 原因 | 替代方式 |
|---|---|---|
| 根目录 `test.py` | 无断言、重复执行一次决策，异常只打印 | `python -m pytest -q`；单次示例用 `main.py --mode decision` |
| `validity_evaluator.py`、`generate_eval_report()`、`--mode eval` / `--report` | 随机生成预测、转化和投诉记录，未调用真实决策链路，容易误读为系统评测 | 回归测试验证实现；真实效果另用标注数据建立评估流程 |
| `run_offline_job()`、`--mode offline` / `--date` | 没有实际计算，只返回完成状态 | 按真实数据管道实现离线任务后再增加入口 |
| 产品加载失败时的默认目录与 API 硬编码列表 | 失败时可能返回与当前数据源无关的演示结果 | 使用实际数据源，失败显式报错，空列表保持为空 |
| 无效配置与依赖 | 字段从未控制对应行为；`.env` 从未自动加载 | 使用 [有效环境变量清单](usage.md#环境变量) |

移除的无效配置包括 `SuitabilityConfig` 及其开关、`enable_auth`、`report_output_dir`、`rate_limit_per_minute`、`redis_db`、`cache_ttl`、日志 `format / max_size_mb / retention_days / enable_audit`，以及只服务这些字段的常量。实际适当性规则、管理令牌保护和现有日志逻辑继续生效；Redis 当前使用默认数据库与现有缓存策略。

不再直接依赖未使用的 `python-dotenv`；测试依赖中的 `httpx` 与运行依赖去重。API / UI 启动器不提前创建决策实例，避免仅启动服务就重复训练模型；UI 通过当前 Python 解释器启动 Streamlit。上述删除涉及旧 Python 方法和 CLI 参数，现有调用方需按表迁移。

## README 设计依据

参考以下项目的文档组织方式，文案与 SVG 示意图为本项目独立编写，没有复制其品牌素材或界面截图：

- [Streamlit](https://github.com/streamlit/streamlit)：简短定位、安装入口和可执行的最小使用流程。
- [Dash](https://github.com/plotly/dash)：状态徽章、功能示例对照和按需深入的文档链接。
- [SHAP](https://github.com/shap/shap)：先解释用途，再连接示例、图形与方法依据。

README 顶部图为概念示意，不是当前看板截图或真实客户测量值。测试徽章链接真实工作流，不静态标注虚构覆盖率、星数或性能提升。版权原文与 LICENSE 保留不变。
