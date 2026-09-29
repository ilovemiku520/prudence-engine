# 多模型 AI 辅助解读

AI 入口位于原客户看板的“AI 辅助解读”页签。它读取已经计算的分数、贡献和因子摘要，不另建一套分析结果，也不更改适当性规则或服务计划。

## 在页面使用

1. 选择模型服务，填写模型名 / 部署名、Base URL 和自己的 API Key。模型名称请从对应平台控制台复制，不绑定固定旧型号。
2. 选择当前群体、所选客户画像或共同因子，填写希望解释的问题。
3. 展开“发送前预览”，核对接收地址、问题和数值摘要，再点击“发送摘要并生成解读”。仅此按钮会发起模型调用。
4. 返回后可阅读或下载文本。切换客户、筛选、利率、产品、问题或模型配置后，不再把旧解读显示为当前结果。

默认发送群体的中位数、四分位数、缺失量、适当性人数及情景金额合计，不发送客户名单。所选客户范围包含该客户的六维分数、原始行为指标和金额，去除姓名和客户 ID。因子范围包含全数据源模型的载荷、共同度和诊断；不发送每个人的因子坐标。去标识摘要仍可能暴露业务信息，小样本尤其如此；你的自定义问题也会发送给所填地址。

页面密钥仅存当前 Streamlit 会话，不写文件、日志、导出内容或共享缓存。切换接收地址会清除该配置的密钥；可点击“清除本次密钥与解读”。不同厂商分别保存会话输入，避免把一个厂商的密钥自动填到另一个厂商。

## 支持的协议

| 协议 | 路径 / 鉴权 | 适用配置 |
|---|---|---|
| openai_responses | `/responses`，Bearer | OpenAI Responses；请求明确 `store=false` |
| openai_chat | `/chat/completions`，Bearer 或 `api-key` | OpenAI 兼容文本接口；DeepSeek、百炼、Ollama；其他网关按其兼容要求配置 |
| anthropic | `/messages`，`x-api-key` + `anthropic-version` | Claude Messages 或兼容网关 |
| gemini | `/models/{model}:generateContent`，`x-goog-api-key` | Gemini Generate Content API |

预设地址包括 OpenAI、DeepSeek、百炼北京地域、Claude、Gemini 与 Ollama。自定义项可以填写符合上述协议的公网 HTTPS 网关；例如 Azure OpenAI **v1** 兼容服务应填写包含 `/openai/v1` 的基础地址，模型字段填写部署名，可选 `api-key` 鉴权。这里没有实现旧版 Azure 带 `api-version` 查询参数的接口、Vertex AI OAuth 或 Bedrock SigV4。

OpenAI 兼容并不意味着每个模型支持完全相同的参数。Chat 模式可选 `max_tokens` 或 `max_completion_tokens`，不强制发送 temperature、工具或结构化输出参数。Claude 和 Gemini 自动使用其专用输出上限字段。正文只取最终回答，不把 reasoning / thinking 块当成解释结果。达到输出限制时显示截断提示；模型只返回思考而无正文时明确提示，不伪造结果。

## 本地模型

由部署者在 Streamlit 进程环境中设置 `AI_ALLOW_LOCAL=true` 后重启，再选择 Ollama，填写本机已安装的模型名。预设为 `http://127.0.0.1:11434/v1`，回环服务可以不填密钥。此处“本机”指运行 Streamlit 的服务器，不是访问网站者的浏览器电脑。

默认禁止访问内网与元数据地址；开启本地选项也只放行明确的回环主机（localhost / 127.0.0.1 / ::1），不开放任意内网。公网域名先解析和检查地址，再固定本次连接 IP，保留原始 Host、TLS SNI 与证书验证；不跟随重定向，不读取系统 HTTP 代理环境变量。需要企业代理时可使用经部署者配置的公网 HTTPS 模型网关。

## 给其他前端的 REST 接口

`POST /api/analysis/explain` 使用现有 `X-Admin-Token` 鉴权，默认关闭。调用此接口表示调用者授权向服务器配置的模型接收方发送计算摘要与问题。调用者不能通过请求体指定上游地址或密钥，避免把服务器做成开放代理。

部署者需向实际运行进程注入环境变量（仅复制 `.env.example` 不会自动加载这些值）：

```text
PRUDENCE_ADMIN_TOKEN=<独立管理令牌>
AI_ENABLED=true
AI_PROTOCOL=openai_chat
AI_BASE_URL=https://api.deepseek.com
AI_MODEL=<平台提供的模型名>
AI_API_KEY=<服务端模型密钥>
AI_MAX_TOKENS=1600
AI_TIMEOUT=60
AI_AUTH_MODE=bearer
AI_TOKEN_PARAMETER=max_tokens
AI_ALLOW_LOCAL=false
```

请求示例（客户/产品 ID 必须存在于 API 的数据源；页面默认的 120 位模拟客户不会自动同步给 API）：

```json
{
  "product_id": "P004",
  "scope": "customer",
  "customer_id": "CUST_HIGH",
  "question": "解释该客户的画像和年度贡献来源，并列出待核实项。",
  "attention_weight": 0.5,
  "scenario": {
    "deposit_rate": 0.015,
    "loan_rate": 0.04,
    "funding_rate": 0.025,
    "wealth_return": 0.035,
    "wealth_fee": 0.005,
    "deposit_share": 0.6,
    "wealth_share": 0.4
  }
}
```

`scope` 可选 overview / customer / factors；`customer_ids` 可选，用于筛选最多 500 个 ID。省略的情景字段使用 ValueScenario 默认值。接口自己读取客户、运行原有评分和情景计算，再构建白名单摘要；不接受客户端伪造的完整分析内容。因子模型使用全数据源拟合，与看板口径一致。

响应含 `text, model, protocol, endpoint, usage, truncated, elapsed_seconds, context_digest, generated_at, advisory_only`。`usage` 仅在服务商返回数字用量时提供，不计算或承诺实际费用。

- 401：管理令牌无效；503：未配置管理令牌或未启用 AI 接口。
- 422：客户/产品、范围、情景或请求字段不合法。
- 502：模型配置、上游鉴权、限流、超时或响应格式错误；不回传上游原始响应或密钥。

REST 接口在工作线程中执行同步模型请求，避免阻塞主事件循环。没有自动重试、自动批量调用或无人值守续费；公网部署仍应在网关配置鉴权、并发限制和费用配额。

## 可复核性与边界

摘要使用实际计算值和明确的公式，缺失保留 null。系统提示要求 AI 引用数值、区分观测与假设、保留硬规则并说明未知；这能约束任务范围，不能保证模型永远正确。AI 输出以纯文本显示，不自动打开链接、加载远程图片、执行代码或触发金融/营销动作。最终数值以看板为准。

不保存跨会话聊天历史，不自动读取整个数据库给模型，不使用模型输出重写分数。统计功能在没有任何 AI Key 时仍可使用。

协议验证采用模拟 HTTP 响应和页面测试，覆盖四种请求格式、鉴权、截断/空响应、错误脱敏、受限地址、稳定快照及明确按钮触发。未使用真实客户数据或真实密钥完成各厂商在线调用；实际模型权限、网络可达性和兼容性需用用户自己的配置验证。

## 官方依据

- [OpenAI：文本生成与 Responses](https://developers.openai.com/api/docs/guides/text)
- [Claude：Messages API](https://platform.claude.com/docs/en/api/messages/create)
- [Gemini：Generate Content](https://ai.google.dev/api/generate-content)
- [DeepSeek：首次 API 调用](https://api-docs.deepseek.com/)
- [百炼：Base URL 总览](https://help.aliyun.com/zh/model-studio/base-url)
- [Ollama：OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)
- [Azure OpenAI：v1 API 与鉴权](https://learn.microsoft.com/en-us/azure/foundry/openai/api-version-lifecycle)

文档核对日期：2026-09-29。实现按这些接口的文本调用协议适配，未声称支持每家平台的全部产品或认证方式。
