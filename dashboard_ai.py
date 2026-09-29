"""Explicit, session-only AI explanation controls inside the customer dashboard."""
from datetime import datetime, timezone
import json
import os
from urllib.parse import urlsplit

import streamlit as st

from ai_explainer import AIError, AISettings, PRESETS, PROTOCOLS, explain
from analysis_context import build_context, context_signature

SCOPES = {"当前客户群体": "overview", "所选客户画像": "customer", "共同因子分析": "factors"}


def clear_endpoint_credentials(key):
    st.session_state.pop(key, None)
    st.session_state.pop("ai_result", None)


def forget_ai():
    for key in list(st.session_state):
        if key.startswith("ai_key_") or key == "ai_result":
            st.session_state.pop(key, None)


def ai_panel(frame, product, scenario, alpha, row, factors, factor_error):
    st.markdown("**AI 辅助解读**")
    st.caption("解释当前看板的计算结果；不会改动评分、规则或服务计划。只有点击发送才会调用模型。")
    provider = st.selectbox("模型服务", list(PRESETS), key="ai_provider")
    default_protocol, default_url = PRESETS[provider]
    tag = str(list(PRESETS).index(provider))
    with st.expander("接口配置", expanded=True):
        a, b = st.columns(2)
        protocol = a.selectbox("接口协议", PROTOCOLS, index=PROTOCOLS.index(default_protocol), key=f"ai_protocol_{tag}")
        model = b.text_input("模型名 / 部署名", key=f"ai_model_{tag}", placeholder="填写服务商控制台提供的准确名称")
        base_url = st.text_input("Base URL", value=default_url, key=f"ai_base_{tag}",
                                 on_change=clear_endpoint_credentials, args=(f"ai_key_{tag}",))
        api_key = st.text_input("API Key", type="password", key=f"ai_key_{tag}",
                               help="仅保留在当前会话。切换接收地址会清除密钥；不写入文件或报告。")
        a, b, c = st.columns(3)
        max_tokens = a.number_input("最多输出 tokens", 128, 8192, 1600, 128, key="ai_max_tokens")
        auth_mode = b.selectbox("兼容接口鉴权", ["bearer", "api-key"], key=f"ai_auth_{tag}",
                                help="Azure v1 可使用 api-key；Claude 与 Gemini 自动使用其专用请求头。")
        token_parameter = c.selectbox("Chat 输出上限字段", ["max_tokens", "max_completion_tokens"], key=f"ai_token_field_{tag}")
        st.caption("支持标准文本接口与自定义网关。模型名由你填写，避免绑定过期型号。Ollama / 本地服务需部署者设置 AI_ALLOW_LOCAL=true。")
        st.button("清除本次密钥与解读", on_click=forget_ai, key="ai_forget")
    scope_label = st.radio("解读范围", list(SCOPES), horizontal=True, key="ai_scope")
    question = st.text_area("希望 AI 解释什么", value="概括主要发现，列出数值依据、局限和下一步值得验证的问题。",
                            max_chars=1500, key="ai_question")
    context = build_context(frame, product, scenario, alpha, scope=SCOPES[scope_label], selected_id=row.customer_id,
                            factors=factors, factor_error=factor_error,
                            synthetic="模拟" in st.session_state.get("source_label", ""))
    settings = AISettings(protocol=protocol, base_url=base_url.strip(), model=model.strip(), api_key=api_key,
                          max_tokens=int(max_tokens), auth_mode=auth_mode, token_parameter=token_parameter,
                          allow_local=os.getenv("AI_ALLOW_LOCAL", "false").lower() == "true")
    local_scope = json.dumps([st.session_state.get("product_id"), st.session_state.get("source_label"),
                             frame.index.tolist(), row.customer_id if SCOPES[scope_label] == "customer" else None])
    signature = context_signature(context, question, settings, local_scope)
    with st.expander("发送前预览 · 接收方与数据", expanded=True):
        st.write("接收地址：", settings.base_url or "尚未填写")
        st.caption("发送下方数值摘要和你的问题，不含姓名、客户 ID、完整客户表或密钥。选择客户画像会包含该客户的金额与评分；去标识摘要仍可能包含敏感业务信息。")
        st.json(dict(question=question, computed_analysis=context), expanded=False)
    try:
        local_without_key = settings.allow_local and urlsplit(settings.base_url).hostname in ("localhost", "127.0.0.1", "::1")
    except ValueError:
        local_without_key = False
    ready = bool(settings.model and settings.base_url and (settings.api_key.strip() or local_without_key))
    if st.button("发送摘要并生成解读", type="primary", disabled=not ready, key="ai_generate"):
        st.session_state.pop("ai_result", None)
        try:
            with st.spinner("正在请求模型解读…"):
                result = explain(settings, context, question)
            result.update(signature=signature, generated_at=datetime.now(timezone.utc).isoformat())
            st.session_state.ai_result = result
        except AIError as error:
            st.error(str(error))
    if not ready:
        st.info("填写模型名和 API Key 后即可生成解读；已开放的本地模型可不填密钥。")
    result = st.session_state.get("ai_result")
    if result and result["signature"] != signature:
        st.info("数据、筛选、客户、问题或模型配置已变化，旧解读不再展示；请重新生成。")
    elif result:
        st.caption(f"{result['model']} · {result['endpoint']} · {result['generated_at']} · {result['elapsed_seconds']} 秒")
        st.info("以下内容由 AI 生成，数值请与看板核对。")
        if result["truncated"]:
            st.warning("回答已达到输出上限，可能不完整；可提高上限后手动重新生成。")
        st.text(result["text"])
        if result["usage"]:
            st.caption("服务商返回的 token 用量：" + json.dumps(result["usage"], ensure_ascii=False))
        report = f"AI 辅助解读\n模型：{result['model']}\n接收方：{result['endpoint']}\n时间：{result['generated_at']}\n\n问题：{question}\n\n{result['text']}\n\n数值以原看板为准。"
        st.download_button("下载本次 AI 解读", report, "ai_analysis.txt", "text/plain", key="ai_download")
