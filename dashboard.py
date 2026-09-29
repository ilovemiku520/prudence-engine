"""睿衡引擎 · 可解释决策与数学分析工作台。"""
from datetime import datetime
from html import escape
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import (matrix_diagnostics, numeric_profile, optimize_allocation,
                       principal_components, pure_nash_equilibria, zero_sum_equilibrium)
from config import get_config
from data_source import build_dataframe_data_source, MockDataSource
from main import PrudenceAPI
from workbench_data import demo_source, read_upload, source_tables

COLORS = ["#0f766e", "#7298c4", "#e8b768", "#b49cc8", "#e78e89"]
LEVELS = {"ALLOW": "可通过", "RESTRICTED": "需复核", "FORBID": "已拦截", "UNKNOWN": "异常"}
ACTIONS = {"PROACTIVE_CLOSING": "主动服务", "NURTURE_CONTENT": "内容培育",
           "LOW_PRIORITY": "暂不打扰", "HUMAN_REVIEW_REQUIRED": "人工复核",
           "BLOCK_AND_REPLACE": "拦截并替代", "ERROR": "决策异常"}
NUMERIC = {"age": "年龄", "assets": "可用资产（元）", "period": "可接受期限（天）",
           "risk_number": "风险等级（序数）", "views": "近 7 日浏览次数",
           "calculator": "计算器使用次数", "duration": "浏览时长"}
PAGES = ["总览", "决策分析", "多元统计", "矩阵实验室", "运筹优化", "博弈实验", "数据管理"]


def style():
    st.markdown("""<style>
    .stApp {font-family: 'Segoe UI','Microsoft YaHei',sans-serif;}
    .block-container {max-width:1440px;padding-top:2.3rem;padding-bottom:3rem;}
    [data-testid="stSidebar"] {background:#edf2f4;border-right:1px solid #dde5e9;}
    [data-testid="stSidebar"] [data-testid="stRadio"] label {padding:9px 12px;margin:2px 0;border-radius:9px;}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {background:#fff;box-shadow:0 2px 6px #163a4910;}
    h1,h2,h3 {letter-spacing:-.035em;}
    h1 {font-weight:750!important;}
    [data-testid="stMetric"] {background:white;border:1px solid #e2e9ec;border-radius:14px;padding:18px 20px;}
    [data-testid="stMetricLabel"] {color:#60727c;font-size:13px;}
    [data-testid="stMetricValue"] {font-size:1.9rem;font-weight:650;color:#193b44;}
    [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {border-radius:12px;overflow:hidden;}
    .hero {padding:30px 34px;border:1px solid #d3e3e2;background:linear-gradient(110deg,#e8f3ef,#f8fafb 70%);border-radius:20px;margin:8px 0 25px;}
    .eyebrow {font-size:11px;letter-spacing:.2em;color:#0f766e;font-weight:700;margin-bottom:14px;}
    .hero h1 {font-size:36px;margin:0 0 10px;line-height:1.25;}
    .hero p {color:#60717a;margin:0;max-width:740px;font-size:14px;line-height:1.9;}
    .brand {font-size:24px;font-weight:750;color:#164b49;margin-bottom:4px;}
    .brand-sub {font-size:10px;letter-spacing:.17em;color:#60757f;margin-bottom:30px;}
    .footer {font-size:11px;color:#798b93;border-top:1px solid #e1e7eb;padding-top:18px;margin-top:30px;}
    .stButton button, .stDownloadButton button {border-radius:9px;}
    @media(max-width:700px){.block-container{padding:1.2rem 1rem}.hero{padding:24px 20px}.hero h1{font-size:27px}}
    </style>""", unsafe_allow_html=True)


def chart(fig, height=340):
    fig.update_layout(template="plotly_white", height=height, paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(family="Segoe UI, Microsoft YaHei", color="#405662", size=12),
                      margin=dict(l=20, r=20, t=54, b=66), colorway=COLORS,
                      legend=dict(orientation="h", y=-.24, x=0, title_text=""),
                      title=dict(x=.035, y=.97, font=dict(size=15, color="#193b44")))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#edf1f4", zeroline=False)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def hero(kicker, title, description):
    st.markdown(f'<div class="hero"><div class="eyebrow">{escape(kicker)}</div>'
                f'<h1>{escape(title)}</h1><p>{escape(description)}</p></div>', unsafe_allow_html=True)


def csv_bytes(frame):
    safe = frame.copy()
    for column in safe.select_dtypes(include=["object", "string"]).columns:
        safe[column] = safe[column].map(
            lambda value: "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value)
    return safe.to_csv(index=False).encode("utf-8-sig")


def download_csv(frame, label, filename):
    st.download_button(label, csv_bytes(frame), filename, "text/csv")


def activate_source(source, label):
    api = PrudenceAPI(get_config(), data_source=source)
    for key in ["batch", "batch_time", "allocation", "history", "import_tables", "capacities"]:
        st.session_state.pop(key, None)
    st.session_state.api = api
    st.session_state.source_label = label


def analysis_frame(api, customers):
    if customers.empty:
        return pd.DataFrame(columns=list(NUMERIC))
    result = customers.set_index("id").copy()
    result["risk_number"] = pd.to_numeric(result.risk.str[1:], errors="coerce")
    signals = [("views", "beh_view_cnt_7d"), ("calculator", "beh_calculator_use_cnt"), ("duration", "beh_view_duration_decay")]
    for label, name in signals:
        result[label] = [api.data_source.get_intent_features(cid).get(name, np.nan) for cid in result.index]
    return result


def risk_matrix(api):
    matrix = api.engines["suitability"].matrix_config
    values = [[matrix.get_cell(f"C{c}", f"R{r}") for r in range(1, 6)] for c in range(1, 6)]
    mapping = {"ALLOW": 0, "RESTRICTED": 1, "FORBID": 2}
    fig = go.Figure(go.Heatmap(z=[[mapping[v] for v in row] for row in values],
        x=[f"R{i}" for i in range(1, 6)], y=[f"C{i}" for i in range(1, 6)],
        text=[[LEVELS[v] for v in row] for row in values], texttemplate="%{text}",
        colorscale=[[0, "#d7eee6"], [.25, "#d7eee6"], [.26, "#f6e6c6"], [.74, "#f6e6c6"], [.75, "#f4d9d7"], [1, "#f4d9d7"]],
        zmin=0, zmax=2, showscale=False, xgap=5, ygap=5,
        hovertemplate="客户 %{y} · 产品 %{x}<br>%{text}<extra></extra>"))
    fig.update_layout(title="适当性基准矩阵", xaxis_title="产品风险", yaxis_title="客户承受能力")
    return fig


def overview(api, customers, products):
    hero("PRUDENCE / OVERVIEW", "让每一个决策，都有依据。",
         "从客户画像出发，连接适当性规则、统计洞察与策略实验。在可解释的分析中，找到更审慎的下一步。")
    columns = st.columns(4)
    columns[0].metric("分析客户", f"{len(customers):,}")
    columns[1].metric("产品池", len(products))
    columns[2].metric("客户资产中位数", f"¥{customers.assets.median() / 10000:,.1f} 万" if not customers.empty else "—")
    columns[3].metric("本次已分析组合", len(st.session_state.get("batch", [])))
    st.write("")
    left, right = st.columns([1.15, 1])
    with left:
        if not customers.empty:
            counts = customers.risk.value_counts().reindex([f"C{i}" for i in range(1, 6)], fill_value=0)
            chart(px.bar(x=counts.index, y=counts.values, title="客户风险结构", labels={"x": "风险等级", "y": "客户数"},
                         color=counts.index, color_discrete_sequence=COLORS).update_layout(showlegend=False))
    with right:
        chart(risk_matrix(api))
        st.caption("基准矩阵仅为首层规则；年龄、首次购买、资产与期限规则还会收紧结果。")
    left, right = st.columns([1.6, 1])
    with left:
        if not customers.empty:
            chart(px.scatter(customers, x="age", y="assets", color="risk", hover_name="id",
                             title="客户画像 · 年龄与可用资产", color_discrete_sequence=COLORS,
                             category_orders={"risk": [f"C{i}" for i in range(1, 6)]},
                             labels={"age": "年龄", "assets": "可用资产（元）", "risk": "风险等级"}))
    with right:
        st.subheader("分析路径")
        for number, title, detail in [("01", "先看数据", "观察分布、相关性与主成分。"),
                                      ("02", "再做决策", "检查适当性，展开意图得分与原因。"),
                                      ("03", "策略实验", "在预算与容量约束下优化服务资源。")]:
            st.markdown(f"**{number} · {title}**")
            st.caption(detail)
        st.info("意图模型为原型。合成训练数据和情景收益不代表真实转化概率或投资回报。")


def decision_page(api, customers, products):
    hero("01 / DECISION INTELLIGENCE", "把决策过程，展开来看。", "先判断适当性，再观察意图信号。支持单个组合与最多 100 个组合的批量对比。")
    if customers.empty or products.empty:
        st.info("请在数据管理中导入客户表和产品表后继续。")
        return
    with st.form("decision_form"):
        left, right = st.columns(2)
        selected_c = left.multiselect("客户", customers.id.tolist(), default=customers.id.head(6).tolist())
        selected_p = right.multiselect("产品", products.id.tolist(), default=products.id.tolist(),
                                       format_func=lambda pid: f"{pid} · {api.get_product(pid).get('name', '')}")
        st.caption("客户数 × 产品数 ≤ 100。禁止组合不会计算意图分，其分数在分析中显示为空。")
        run = st.form_submit_button("运行决策分析", type="primary")
    if run:
        if not selected_c or not selected_p or len(selected_c) * len(selected_p) > 100:
            st.error("请至少选择 1 位客户和 1 个产品，并将总组合数控制在 100 以内。")
        else:
            with st.spinner("正在逐项检查适当性与意图…"):
                results = api.batch_decide([{"customer_id": c, "product_id": p} for c in selected_c for p in selected_p])
            st.session_state.batch = results
            st.session_state.batch_time = datetime.now().isoformat(timespec="seconds")
            st.session_state.pop("allocation", None)
            history = st.session_state.get("history", [])
            history.append({"time": st.session_state.batch_time, "results": results})
            st.session_state.history = history[-10:]
    results = st.session_state.get("batch", [])
    if not results:
        st.info("选好分析范围后运行，结果会保留在当前会话，并可用于运筹优化。")
        return
    st.caption(f"结果快照 · {st.session_state.get('batch_time', '')} · {len(results)} 个组合（修改选择后需重新运行）")
    df = pd.DataFrame(results)
    for score in ["intent_score", "rule_score", "model_score"]:
        if score not in df:
            df[score] = np.nan
        df.loc[df.suitability_level.isin(["FORBID", "UNKNOWN"]), score] = np.nan
    cols = st.columns(4)
    for col, level in zip(cols[:3], ["ALLOW", "RESTRICTED", "FORBID"]):
        col.metric(LEVELS[level], int(df.suitability_level.eq(level).sum()))
    cols[3].metric("异常组合", int(df.action.eq("ERROR").sum()))
    chart_tab, detail_tab, report_tab = st.tabs(["可视分析", "决策明细", "报告与历史"])
    with chart_tab:
        left, right = st.columns(2)
        with left:
            counts = df.suitability_level.map(LEVELS).value_counts()
            chart(px.pie(names=counts.index, values=counts.values, hole=.7, title="适当性分布", color=counts.index,
                         color_discrete_map={"可通过": COLORS[0], "需复核": COLORS[2], "已拦截": COLORS[4], "异常": "#87929b"}))
        with right:
            chart(px.scatter(df.dropna(subset=["rule_score", "model_score"]), x="rule_score", y="model_score",
                             color="suitability_level", hover_data=["customer_id", "product_id"], title="规则分 × 模型分",
                             range_x=[0, 1], range_y=[0, 1], color_discrete_sequence=COLORS,
                             labels={"rule_score": "规则分", "model_score": "模型分", "suitability_level": "适当性"}))
        pivot = df.pivot(index="customer_id", columns="product_id", values="intent_score")
        chart(px.imshow(pivot, text_auto=".2f", zmin=0, zmax=1, color_continuous_scale="Teal", aspect="auto",
                        title="客户 × 产品意图分", labels={"color": "意图分"}), min(640, max(300, len(pivot) * 28)))
        st.caption("空白表示未计算。同一客户在不同产品上的分数可能相同，因为当前模型主要使用客户行为特征。")
    with detail_tab:
        display = df[["customer_id", "product_id", "suitability_level", "action", "intent_score", "rule_score", "model_score", "reason"]]
        st.dataframe(display, use_container_width=True, hide_index=True, column_config={
            "customer_id": "客户", "product_id": "产品", "suitability_level": "适当性",
            "action": "动作", "intent_score": st.column_config.NumberColumn("意图分", format="%.3f"),
            "rule_score": st.column_config.NumberColumn("规则分", format="%.3f"),
            "model_score": st.column_config.NumberColumn("模型分", format="%.3f"), "reason": "原因"})
        selected = st.selectbox("展开组合", range(len(results)),
                                format_func=lambda i: f"{results[i]['customer_id']} → {results[i]['product_id']}")
        record = results[selected]
        st.info(f"{ACTIONS.get(record['action'], record['action'])} · {record.get('reason', '')}")
        left, right = st.columns(2)
        left.json(api.get_customer(record["customer_id"]), expanded=True)
        right.json(api.get_product(record["product_id"]), expanded=True)
        signals = record.get("top_signals", [])
        if signals:
            chart(px.bar(pd.DataFrame(signals), x="shap_value", y="feature", orientation="h", title="主要 SHAP 信号（模型贡献）"), 250)
            st.caption("SHAP 描述模型贡献，不能解释因果。")
        else:
            st.caption("该组合没有可用的 SHAP 解释。禁止组合不会进入意图评分。")
        if record.get("replacement_products"):
            st.write("替代候选产品（仍需逐项运行适当性检查）")
            st.dataframe(pd.DataFrame(record["replacement_products"]), hide_index=True)
    with report_tab:
        report_frame = df.drop(columns=["top_signals", "replacement_products"], errors="ignore")
        download_csv(report_frame, "下载决策 CSV", "prudence_decisions.csv")
        payload = {"source": st.session_state.source_label, "generated_at": st.session_state.batch_time,
                   "note": "原型决策；FORBID/ERROR 分数为未计算。", "results": json.loads(df.to_json(orient="records", force_ascii=False))}
        st.download_button("下载完整 JSON", json.dumps(payload, ensure_ascii=False, indent=2), "prudence_report.json", "application/json")
        html = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>睿衡分析报告</title>"
        html += "<style>body{font-family:sans-serif;margin:40px;color:#193b44}table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border:1px solid #dde5e9}</style>"
        html += f"<h1>睿衡 · 决策报告</h1><p>{escape(st.session_state.source_label)} · {escape(st.session_state.batch_time)}</p>"
        html += "<p>教学原型；模拟数据不代表实际业务表现。空分数代表未计算。</p>"
        html += report_frame.to_html(index=False, escape=True, na_rep="未计算") + "</html>"
        st.download_button("下载 HTML 报告", html, "prudence_report.html", "text/html")
        history = st.session_state.get("history", [])
        if history:
            index = st.selectbox("本会话最近 10 次分析", range(len(history)), format_func=lambda i: f"{history[i]['time']} · {len(history[i]['results'])} 个组合")
            if st.button("恢复此分析快照"):
                st.session_state.batch = history[index]["results"]
                st.session_state.batch_time = history[index]["time"]
                st.session_state.pop("allocation", None)
                st.rerun()


def statistics_page(api, customers):
    hero("02 / MULTIVARIATE ANALYSIS", "从多个维度，理解客户。", "以一位客户为一条观测，探索变量分布与相关性，再用标准化 PCA 压缩信息。")
    frame = analysis_frame(api, customers)
    fields = st.multiselect("分析字段", list(NUMERIC), default=list(NUMERIC)[:5], format_func=NUMERIC.get)
    if not fields or frame.empty:
        st.info("请准备客户数据，并至少选择一个字段。")
        return
    numeric, summary = numeric_profile(frame[fields].rename(columns=NUMERIC))
    cols = st.columns(3)
    cols[0].metric("客户样本量", len(numeric))
    cols[1].metric("完整样本", len(numeric.dropna()))
    cols[2].metric("缺失 / 无效单元格", int(numeric.isna().sum().sum()))
    if len(numeric) < 20:
        st.warning("当前样本较少，图表只用于描述这组样本，不宜推广为总体结论。")
    distribution, correlation, pca = st.tabs(["分布与质量", "相关性", "PCA 主成分"])
    with distribution:
        selected = st.selectbox("观察变量", numeric.columns)
        left, right = st.columns(2)
        with left:
            chart(px.histogram(numeric, x=selected, nbins=20, title="样本分布", color_discrete_sequence=COLORS))
        with right:
            chart(px.box(numeric, y=selected, points="outliers", title="中位数、四分位数与异常点", color_discrete_sequence=COLORS))
        st.dataframe(summary, use_container_width=True, column_config={
            "count": "有效数", "mean": "均值", "std": "标准差", "min": "最小值", "max": "最大值",
            "missing": "缺失数", "missing_rate": st.column_config.NumberColumn("缺失率", format="%.2f")})
        download_csv(summary.reset_index(names="字段"), "下载描述统计", "descriptive_statistics.csv")
    with correlation:
        method = st.radio("相关系数", ["spearman", "pearson"], horizontal=True,
                          format_func=lambda m: "Spearman 秩相关" if m == "spearman" else "Pearson 线性相关")
        chart(px.imshow(numeric.corr(method=method, min_periods=3), text_auto=".2f", zmin=-1, zmax=1,
                        color_continuous_scale="Tealrose", title="相关矩阵 · 成对完整样本"), 430)
        with st.expander("查看每对变量实际使用的样本数"):
            present = numeric.notna().astype(int)
            st.dataframe(present.T @ present, use_container_width=True)
        st.caption("风险等级属于序数变量，优先查看秩相关。常量列或样本不足的系数留空；相关不代表因果。")
    with pca:
        try:
            result = principal_components(numeric)
        except ValueError as error:
            st.info(str(error))
            return
        st.caption(f"标准化：减均值 / 样本标准差 · 排除 {result['dropped_rows']} 行缺失样本 · 去除常量字段：{', '.join(result['constant_columns']) or '无'}")
        variance = result["variance_ratio"]
        left, right = st.columns(2)
        with left:
            fig = go.Figure(go.Bar(x=variance.index, y=variance, name="单项解释率", marker_color=COLORS[0]))
            fig.add_scatter(x=variance.index, y=variance.cumsum(), name="累计解释率", mode="lines+markers", line_color=COLORS[2])
            fig.update_layout(title="主成分解释方差", yaxis=dict(tickformat=".0%", range=[0, 1.05]))
            chart(fig)
        with right:
            scores = result["scores"].copy()
            scores["risk"] = frame.loc[scores.index, "risk"]
            scores["客户"] = scores.index
            chart(px.scatter(scores, x="PC1", y="PC2", color="risk", hover_name="客户", color_discrete_sequence=COLORS,
                             title=f"二维投影 · 保留 {variance.iloc[:2].sum():.1%} 的样本方差", labels={"risk": "客户风险"}))
        chart(px.imshow(result["loadings"].iloc[:, :min(4, len(variance))], text_auto=".2f", color_continuous_scale="Tealrose",
                        zmin=-1, zmax=1, aspect="auto", title="载荷矩阵 · 变量与主成分的相关程度"), 360)
        st.caption("PCA 使用完整样本与样本标准差，通过 SVD 计算。载荷 = 特征向量 × 特征值平方根；方向正负无优劣，二维投影不是客户评级。")
        download_csv(result["scores"].reset_index(), "下载主成分得分", "pca_scores.csv")


def matrix_page(api, customers):
    hero("03 / LINEAR ALGEBRA", "看见矩阵背后的结构。", "检查秩、零空间维数与条件数；通过奇异值分解，观察低秩近似保留了多少信息。")
    mode = st.radio("矩阵来源", ["客户标准化特征", "自定义矩阵"], horizontal=True)
    if mode == "客户标准化特征":
        try:
            result = principal_components(analysis_frame(api, customers)[["age", "assets", "period", "risk_number"]])
            a = result["standardized"].to_numpy()
        except ValueError as error:
            st.info(str(error))
            return
        st.caption(f"{a.shape[0]} 位客户 × {a.shape[1]} 个字段；已使用样本标准差标准化。")
    else:
        text = st.text_area("每行一组数字，以空格或逗号分隔（最多 20 × 20）", "3, 1, 0\n1, 3, 0\n0, 0, 1", height=120)
        try:
            a = np.array([[float(value) for value in line.replace(",", " ").split()] for line in text.strip().splitlines()])
            if a.ndim != 2 or 0 in a.shape or max(a.shape) > 20:
                raise ValueError()
        except ValueError:
            st.error("请填写每行长度相同的数值矩阵，维度不超过 20 × 20。")
            return
    k = st.select_slider("近似阶数 k", options=list(range(1, min(a.shape) + 1)))
    try:
        result = matrix_diagnostics(a, k)
    except ValueError as error:
        st.error(str(error))
        return
    cols = st.columns(4)
    cols[0].metric("矩阵秩", result["rank"])
    cols[1].metric("零空间维数", result["nullity"])
    cols[2].metric("条件数 κ₂", f"{result['condition']:,.2f}" if np.isfinite(result["condition"]) else "∞")
    cols[3].metric("相对重建误差", f"{result['relative_error']:.2%}")
    left, right = st.columns(2)
    with left:
        chart(px.bar(x=[f"σ{i+1}" for i in range(len(result["singular_values"]))], y=result["singular_values"], title="奇异值谱"))
    with right:
        chart(px.line(x=list(range(1, len(result["energy"]) + 1)), y=result["energy"], markers=True,
                      title="累计能量保留", labels={"x": "近似阶数", "y": "保留比例"}).update_yaxes(tickformat=".0%"))
    st.latex(r"A = U\Sigma V^T,\quad A_k = U_k\Sigma_k V_k^T,\quad \epsilon_k=\frac{\|A-A_k\|_F}{\|A\|_F}")
    if not np.isfinite(result["condition"]) or result["condition"] > 1e8:
        st.warning("矩阵秩亏或条件数很高，直接求逆可能不稳定；分析采用 SVD，未强行求逆。")
    if result["eigenvalues"] is not None:
        st.write("对称矩阵特征值", result["eigenvalues"].tolist())
    with st.expander("查看近似矩阵（前 50 行）"):
        st.dataframe(result["reconstructed"][:50], use_container_width=True)
    download_csv(pd.DataFrame(result["reconstructed"]), "下载近似矩阵", "matrix_approximation.csv")


def optimization_page():
    hero("04 / OPERATIONS RESEARCH", "把有限资源，用在合适的组合上。", "使用 0–1 整数规划，在服务预算、触达人数和产品容量的约束下，最大化情景效用分。")
    results = st.session_state.get("batch", [])
    if not results:
        st.info("先到「决策分析」运行一组客户与产品，适当性结果会成为此处的硬约束。")
        return
    base = pd.DataFrame(results)
    allowed = base.loc[base.suitability_level.eq("ALLOW")]
    st.info(f"当前 {len(base)} 个组合中，仅 {len(allowed)} 个 ALLOW 组合可分配。RESTRICTED、FORBID 与异常组合全部排除；每位客户最多 1 个产品。")
    if allowed.empty:
        return
    a, b, c = st.columns(3)
    budget = a.number_input("总服务预算（成本单位）", min_value=0.0, value=120.0, step=10.0)
    contacts = b.number_input("最多触达客户数", min_value=0, max_value=100, value=min(6, allowed.customer_id.nunique()))
    unit_cost = c.number_input("每次服务成本（成本单位）", min_value=0.0, value=20.0, step=1.0)
    capacities = pd.DataFrame({"产品": sorted(allowed.product_id.unique()), "容量": 3})
    edited = st.data_editor(capacities, disabled=["产品"], hide_index=True, use_container_width=True,
                            column_config={"容量": st.column_config.NumberColumn(min_value=0, max_value=100, step=1)}, key="capacities")
    st.caption("情景效用默认取意图分 × 100，只表示模型排序偏好；服务成本由用户设定，不是产品购买金额。")
    candidates = base[["customer_id", "product_id", "suitability_level", "intent_score"]].copy()
    candidates["utility"] = candidates.intent_score * 100
    candidates["cost"] = unit_cost
    signature = (st.session_state.batch_time, float(budget), int(contacts), float(unit_cost), edited.to_json())
    if st.button("求解最优分配", type="primary"):
        try:
            capacities_map = dict(zip(edited["产品"], edited["容量"]))
            result = optimize_allocation(candidates, budget, contacts, capacities_map)
            curve = []
            for limit in sorted(set([0, budget * .25, budget * .5, budget * .75, budget, budget * 1.25])):
                point = optimize_allocation(candidates, limit, contacts, capacities_map)
                curve.append({"服务预算": limit, "最优情景效用": point.objective, "分配人数": len(point.selected)})
            st.session_state.allocation = (signature, result, pd.DataFrame(curve))
        except (ValueError, TypeError) as error:
            st.error(str(error))
    saved = st.session_state.get("allocation")
    if saved and saved[0] != signature:
        st.info("参数已改变，请重新求解以更新结果。")
    elif saved:
        _, result, curve = saved
        a, b, c = st.columns(3)
        a.metric("最优情景效用", f"{result.objective:.2f}")
        b.metric("已分配客户", len(result.selected))
        c.metric("预算使用", f"{result.cost:.0f} / {budget:.0f}")
        st.dataframe(result.selected, hide_index=True, use_container_width=True)
        if result.selected.empty:
            st.info("当前约束下空方案最优：预算、容量或效用不足，未分配任何组合。")
        chart(px.line(curve, x="服务预算", y="最优情景效用", markers=True, title="预算敏感性 · 其余约束保持不变"))
        download_csv(result.selected, "下载分配方案", "allocation.csv")
    with st.expander("模型定义"):
        st.latex(r"\max\sum_{i,j}u_{ij}x_{ij}\quad s.t.\quad \sum_j x_{ij}\leq1,\ \sum_i x_{ij}\leq K_j,\ \sum_{i,j}c_{ij}x_{ij}\leq B,\ \sum_{i,j}x_{ij}\leq N")
        st.caption("x 为 0/1；非 ALLOW 组合固定为 0。使用 HiGHS 整数规划求解，仅在求解器确认最优时展示方案。")


def game_page():
    hero("05 / GAME THEORY", "在策略互动中，理解均衡。", "编辑收益矩阵，观察双方最佳回应。这里的收益是人为设定的实验分值，与理财产品回报无关。")
    mode = st.radio("实验类型", ["零和博弈 · 混合策略", "双矩阵博弈 · 纯策略 Nash"], horizontal=True)
    if mode.startswith("零和"):
        preset = st.selectbox("示例", ["策略轮换（石头剪刀布）", "匹配硬币", "存在鞍点"])
        matrices = {"策略轮换（石头剪刀布）": [[0, -1, 1], [1, 0, -1], [-1, 1, 0]],
                    "匹配硬币": [[1, -1], [-1, 1]], "存在鞍点": [[3, 1], [4, 2]]}
        values = matrices[preset]
        frame = pd.DataFrame(values, index=[f"行策略 {i+1}" for i in range(len(values))], columns=[f"列策略 {i+1}" for i in range(len(values[0]))], dtype=float)
        a = st.data_editor(frame, key=f"zero_{preset}", use_container_width=True)
        st.caption("表中为行玩家收益 A，列玩家收益为 −A；行玩家最大化，列玩家最小化。")
        try:
            result = zero_sum_equilibrium(a)
        except ValueError as error:
            st.error(str(error))
            return
        columns = st.columns(3)
        columns[0].metric("均衡博弈值", f"{result['value']:.4f}")
        columns[1].metric("纯策略保底收益", f"{result['pure_maximin']:.4f}")
        columns[2].metric("对偶间隙", f"{result['duality_gap']:.2e}")
        left, right = st.columns(2)
        with left:
            chart(px.bar(x=a.index, y=result["row_strategy"], title="行玩家 · 最优混合策略", labels={"x": "策略", "y": "概率"}).update_yaxes(range=[0, 1], tickformat=".0%"))
        with right:
            chart(px.bar(x=a.columns, y=result["column_strategy"], title="列玩家 · 最优混合策略", labels={"x": "策略", "y": "概率"}).update_yaxes(range=[0, 1], tickformat=".0%"))
        st.latex(r"\max_{p\in\Delta_m}\min_j(p^T A)_j = \min_{q\in\Delta_n}\max_i(Aq)_i")
        st.caption("求解双方线性规划，并用最佳回应收益差检验结果。均衡可能不唯一，这里展示其中一组。")
    else:
        left, right = st.columns(2)
        with left:
            st.write("行玩家收益")
            a = st.data_editor(pd.DataFrame([[3., 0.], [5., 1.]], index=["合作", "背离"], columns=["合作", "背离"]), key="nash_a")
        with right:
            st.write("列玩家收益")
            b = st.data_editor(pd.DataFrame([[3., 5.], [0., 1.]], index=["合作", "背离"], columns=["合作", "背离"]), key="nash_b")
        try:
            equilibria = pure_nash_equilibria(a, b)
        except ValueError as error:
            st.error(str(error))
            return
        if equilibria:
            st.success("纯策略 Nash 均衡：" + "；".join(f"行玩家 {a.index[i]} / 列玩家 {a.columns[j]}（收益 {a.iloc[i,j]:g}, {b.iloc[i,j]:g}）" for i, j in equilibria))
        else:
            st.info("不存在纯策略 Nash 均衡；这不代表不存在混合策略均衡。本模式未求解一般双矩阵混合均衡。")
        st.caption("纯策略 Nash：另一方策略固定时，任一方单独改变策略都不能获得更高收益。共同利益最高的方案未必是均衡。")


def data_page(api, customers, products):
    hero("06 / DATA WORKSPACE", "可信分析，从清楚的数据开始。", "支持 CSV、Excel 和 JSON。先解析、检查并预览，再明确应用；更换数据源后会清除旧分析快照。")
    left, right = st.columns(2)
    if left.button("加载 120 位模拟客户", use_container_width=True):
        activate_source(demo_source(), "模拟数据 · 120 位客户 · seed 42")
        st.rerun()
    if right.button("恢复原版 3 位演示客户", use_container_width=True):
        activate_source(MockDataSource(), "模拟数据 · 原版 3 位客户")
        st.rerun()
    st.caption("模拟样本由固定随机种子生成，便于复现。数据不会自动写入数据库。")
    with st.form("upload_form"):
        uploaded = st.file_uploader("导入文件", type=["csv", "xlsx", "json"])
        submitted = st.form_submit_button("解析并预览")
    if submitted:
        if uploaded is None:
            st.warning("请先选择文件。")
        else:
            try:
                parsed = read_upload(uploaded.name, uploaded.getvalue())
                staged = st.session_state.get("import_tables", {}).copy()
                staged.update(parsed)
                st.session_state.import_tables = staged
                st.success("已加入待导入区，可继续上传其他表。尚未更换当前数据源。")
            except (ValueError, KeyError, TypeError, UnicodeError, OSError) as error:
                st.error(f"导入失败：{error}")
    staged = st.session_state.get("import_tables", {})
    if staged:
        for name, frame in staged.items():
            st.write(f"待导入 · {name} · {len(frame)} 行")
            st.dataframe(frame.head(10), hide_index=True, use_container_width=True)
        a, b = st.columns(2)
        if a.button("应用待导入数据", type="primary"):
            try:
                if any(key not in staged or staged[key].empty for key in ["customers", "products"]):
                    raise ValueError("请先上传非空客户表和产品表。可以分次上传 CSV，或一次上传完整 JSON / Excel。")
                source = build_dataframe_data_source(staged["customers"], staged["products"], staged.get("intent_features"))
                activate_source(source, "用户导入数据")
                st.rerun()
            except ValueError as error:
                st.error(str(error))
        if b.button("清空待导入区"):
            st.session_state.pop("import_tables", None)
            st.rerun()
    with st.expander("当前客户与产品"):
        st.dataframe(customers, hide_index=True, use_container_width=True)
        st.dataframe(products, hide_index=True, use_container_width=True)
    sample = demo_source(size=5)
    sample_c, sample_p = source_tables(sample)
    payload = {"customers": sample_c.to_dict("records"), "products": sample_p.to_dict("records"),
               "intent_features": [{"customer_id": cid, "feature_name": name, "feature_value": value}
                                   for cid in sample.list_customers() for name, value in sample.get_intent_features(cid).items()]}
    st.download_button("下载完整 JSON 示例", json.dumps(payload, ensure_ascii=False, indent=2), "prudence_sample.json", "application/json")
    with st.expander("字段说明与计算口径"):
        st.markdown("客户：`id, risk, age, assets, period, first_buy`。产品：`id, risk, name, lock, min`。意图：`customer_id, feature_name, feature_value`。")
        st.markdown("风险等级为 C1–C5 / R1–R5；金额和期限不能为负；标识不能重复。`false` 按布尔假解析。缺失意图特征按 0 补齐用于模型计算；统计仍将未提供的行为值显示为缺失。")
        st.markdown("统计以客户为独立观测，避免将同一客户的多个产品重复当成样本。PCA 使用完整样本；所有分数、效用和博弈结果都应按各自口径解释。")


def main():
    st.set_page_config(page_title="睿衡 · 决策分析工作台", page_icon="◈", layout="wide")
    style()
    if "api" not in st.session_state:
        with st.spinner("正在准备工作台，首次启动需要训练演示模型…"):
            config = get_config()
            if config.data_source.type == "mock":
                activate_source(demo_source(), "模拟数据 · 120 位客户 · seed 42")
            else:
                st.session_state.api = PrudenceAPI(config)
                st.session_state.source_label = f"配置数据源 · {config.data_source.type}"
    api = st.session_state.api
    customers, products = source_tables(api.data_source)
    with st.sidebar:
        st.markdown('<div class="brand">◈ 睿衡引擎</div><div class="brand-sub">PRUDENCE / ANALYTICS STUDIO</div>', unsafe_allow_html=True)
        page = st.radio("工作空间", PAGES, key="navigation", label_visibility="collapsed")
        st.divider()
        st.caption("当前数据源")
        st.write(st.session_state.source_label)
        st.caption(f"{len(customers)} 位客户 · {len(products)} 个产品")
        st.divider()
        st.caption("方法可解释 · 规则优先 · 结果可导出")
        st.caption("教学与作品集原型，需结合机构规则与人工审核使用。")
    if page == "总览":
        overview(api, customers, products)
    elif page == "决策分析":
        decision_page(api, customers, products)
    elif page == "多元统计":
        statistics_page(api, customers)
    elif page == "矩阵实验室":
        matrix_page(api, customers)
    elif page == "运筹优化":
        optimization_page()
    elif page == "博弈实验":
        game_page()
    else:
        data_page(api, customers, products)
    st.markdown('<div class="footer">PRUDENCE ENGINE · 睿衡分析工作台　/　探索数据，审慎决策。</div>', unsafe_allow_html=True)
