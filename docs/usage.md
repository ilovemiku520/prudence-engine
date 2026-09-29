# 使用与配置

[返回 README](../README.md) · [AI 接入](ai-analysis.md) · [计算方法](analytics-methods.md)

## 页面与数据

`python -m streamlit run ui.py` 启动统一看板，默认使用 seed=42 的 120 位模拟客户。散点点选对应右侧画像；可搜索同位置客户，也可框选或套索形成服务计划候选。

左侧“利率与理财策略”修改后须点击“应用利率与策略”。这些参数只影响当前会话，不写回数据库、产品表或 REST API 的配置。六维观察评分不使用合成标签训练的意图模型。

“数据与导入”支持先预览、后应用 CSV / Excel / JSON。CSV 可分次上传客户、产品和特征表；Excel 可用多个工作表；JSON 的顶层键为 `customers`、`products`、`intent_features`。客户与产品表必需，特征表可选；缺少行为特征时画像保留缺失，综合观察分可能不可计算。页面提供完整 JSON 示例下载。

### 客户表

```csv
id,risk,age,assets,period,first_buy,name,income,loan_balance
CUST_001,C3,35,800000,365,false,示例客户,中,0
```

必需字段：`id, risk, age, assets, period, first_buy`。`risk` 为 C1–C5；金额为元，期限为天。`name, income, loan_balance` 可选。贷款空值会使用情景假设，真实 `0` 保持零余额。

### 产品表

```csv
id,risk,name,lock,min,type
P001,R1,示例货币产品,0,0,货币型
```

必需字段：`id, risk, name, lock, min`；`risk` 为 R1–R5，锁定期限为天，起投额为元。标识不能重复，金额和期限不能为负。

### 行为特征表

```csv
customer_id,feature_name,feature_value
CUST_001,beh_calculator_use_cnt,3
CUST_001,beh_view_cnt_7d,8
```

其他画像特征及锚点见 [计算方法](analytics-methods.md)。客户是独立观测单位，不把同一客户的多个产品重复计作多位客户。

## 环境变量

### 安装范围

统一看板仅需 `python -m pip install -r requirements.txt`。独立 CLI 决策和 REST API 还需执行 `python -m pip install -r requirements-engine.txt`，其中包括预测模型、SHAP 和服务依赖。`requirements-dev.txt` 会安装完整依赖供测试使用。模型与 Redis 环境变量仅影响独立决策链路，不影响看板启动。

`.env.example` 是配置清单，应用**不自动读取 `.env` 文件**。请在启动进程前由终端、容器或托管平台注入变量。统计看板不需要模型密钥；个人 AI 配置可直接在页面填写。

| 变量 | 默认值 / 用途 |
|---|---|
| `DS_TYPE` | `mock`；可选 `sqlite, mysql, postgresql`，关系数据库需提供相应表结构 |
| `DS_DB_PATH` | SQLite 路径，默认 `./prudence.db`；该适配器会初始化演示表与样本 |
| `DS_HOST, DS_PORT, DS_DATABASE, DS_USER, DS_PASSWORD` | 数据库连接；默认端口 3306，PostgreSQL 请明确设置实际端口 |
| `INTENT_MODEL_PATH` | `./intent_model.pkl`；缺少模型时用合成数据训练，仅作原型演示 |
| `INTENT_STAGE` | `stable`，意图融合阶段 |
| `INTENT_THRESHOLD_HIGH, INTENT_THRESHOLD_LOW` | `0.7 / 0.4`，独立决策接口的阈值 |
| `REDIS_ENABLE, REDIS_HOST, REDIS_PORT` | 默认关闭；启用后使用 Redis 的默认数据库，无 TTL 配置；不可用时回退内存 |
| `API_HOST, API_PORT, API_DEBUG` | `0.0.0.0 / 8000 / false`；本地使用可将 host 设为 `127.0.0.1` |
| `API_CORS_ORIGINS` | 逗号分隔，默认本地 Streamlit 的两个 8501 地址 |
| `PRUDENCE_ADMIN_TOKEN` | 默认空；未设置时客户、审计、指标和 AI 管理接口关闭 |
| `PRUDENCE_AUDIT_SALT` | 客户/产品标识哈希所用盐；部署时配置独立随机值 |
| `LOG_LEVEL, LOG_DIR` | `INFO / ./logs` |
| `AI_*` | 见 [AI 接入说明](ai-analysis.md)，REST 解读默认关闭 |

PowerShell 示例：

```powershell
$env:API_HOST = "127.0.0.1"
$env:API_PORT = "8000"
python api.py
```

macOS / Linux 示例：

```bash
export API_HOST=127.0.0.1
export API_PORT=8000
python api.py
```

不要提交真实 `.env`、密钥、数据库、客户表或模型文件。页面 AI 密钥与服务端 AI 密钥分别配置，页面不自动取用服务端密钥。

## 命令行与 REST

单次与批量决策保留为独立调用入口，默认使用 `MockDataSource` 的 3 位示例客户，**不是看板的 120 位样本**：

```bash
python main.py --mode decision --customer CUST_HIGH --product P004
echo '[{"customer_id":"CUST_HIGH","product_id":"P004"}]' | python main.py --mode batch
python main.py --mode api
```

JSON 配置文件用于 `decision / batch / api`，通过 `--config path/to/config.json` 指定；会从文件构建配置，不再与环境变量合并。显式给出不存在的配置文件会报错。UI 启动可使用 `python main.py --mode ui --port 8501`，配置通过其继承的进程环境传入。

默认 API 文档地址：`http://127.0.0.1:8000/api/docs`。

| 接口 | 用途 | 访问控制 |
|---|---|---|
| `GET /api/health`（或 `/`） | 服务健康状态 | 无应用内鉴权 |
| `POST /api/decision` | 单次融合决策 | 无应用内鉴权 |
| `POST /api/decision/batch` | 最多 100 条批量决策 | 无应用内鉴权 |
| `GET /api/products`、`GET /api/product/{id}` | 产品查询 | 无应用内鉴权 |
| `GET /api/customers`、`GET /api/customer/{id}` | 客户查询 | `X-Admin-Token` |
| `GET /api/metrics`、`GET /api/audit` | 进程内指标与审计 | `X-Admin-Token` |
| `POST /api/analysis/explain` | 服务器计算摘要后调用 AI | `X-Admin-Token`，且须开启 `AI_ENABLED` |

这些接口的应用内保护范围并不等于完整的生产身份系统。公网部署须在网关配置身份认证、TLS、限流和费用配额。旧版 `enable_auth` / `rate_limit_per_minute` 从未接入请求链路，现已移除，不能靠设置它们获得保护。

数据源为空时列表保持为空；未知详情返回 404。产品读取失败会显式失败，不用内置演示目录替换实际数据。
