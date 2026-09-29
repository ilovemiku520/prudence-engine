"""Shared presentation and data import helpers for the integrated customer dashboard."""
from html import escape
import json
import pandas as pd
import streamlit as st
from config import get_config
from data_source import build_dataframe_data_source, MockDataSource
from main import PrudenceAPI
from workbench_data import demo_source, read_upload, source_tables
COLORS = ["#0f766e", "#7298c4", "#e8b768", "#b49cc8", "#e78e89"]

def style():
    st.markdown("""<style>
    .stApp {font-family: 'Segoe UI','Microsoft YaHei',sans-serif;}
    .block-container {max-width:1440px;padding-top:1.4rem;padding-bottom:3rem;}
    [data-testid="stSidebar"] {background:#edf2f4;border-right:1px solid #dde5e9;}
    [data-testid="stSidebar"] [data-testid="stRadio"] label {padding:9px 12px;margin:2px 0;border-radius:9px;}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {background:#fff;box-shadow:0 2px 6px #163a4910;}
    h1,h2,h3 {letter-spacing:-.035em;}
    h1 {font-weight:750!important;}
    [data-testid="stMetric"] {background:white;border:1px solid #e2e9ec;border-radius:14px;padding:13px 16px;}
    [data-testid="stMetricLabel"] {color:#60727c;font-size:13px;}
    [data-testid="stMetricValue"] {font-size:1.9rem;font-weight:650;color:#193b44;}
    [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {border-radius:12px;overflow:hidden;}
    .hero {padding:20px 26px;border:1px solid #d3e3e2;background:linear-gradient(110deg,#e8f3ef,#f8fafb 70%);border-radius:20px;margin:8px 0 18px;}
    .eyebrow {font-size:11px;letter-spacing:.2em;color:#0f766e;font-weight:700;margin-bottom:14px;}
    .hero h1 {font-size:29px;margin:0 0 8px;line-height:1.25;padding:0!important;}
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
    for key in list(st.session_state):
        if key not in {"api", "source_label"}:
            st.session_state.pop(key, None)
    st.session_state.api = api
    st.session_state.source_label = label


def data_page(api, customers, products):
    st.caption("CSV / Excel / JSON：先解析预览，再应用。切换数据会清除旧情景与选择。")
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
        st.markdown("风险等级为 C1–C5 / R1–R5；金额和期限不能为负；标识不能重复。`false` 按布尔假解析。画像评分保留缺失值，不用 0 替代；不完整客户仍可查看画像与价值情景。可选 loan_balance 为贷款余额（元）。")
        st.markdown("统计以客户为独立观测，避免将同一客户的多个产品重复当成样本。因子分析使用完整样本；观察评分、情景贡献和潜在因子各有独立口径。")


