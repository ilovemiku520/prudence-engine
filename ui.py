# ui.py
"""
睿衡引擎 - Streamlit 可视化仪表板
支持：数据源配置（模拟/文件上传/数据库）、客户/产品选择、决策执行、历史记录、导出报告
新增：多人数据对比模式（多客户、多产品批量决策与可视化对比）
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import json
from typing import Dict, Optional, List
import io

from main import PrudenceAPI
from config import get_config
from data_source import build_dataframe_data_source, MemoryDataSource, DataSource

# ================================================================
# 1. 页面配置
# ================================================================

st.set_page_config(
    page_title="睿衡引擎 · 决策仪表板",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚖️ 睿衡引擎 · 适当性与意图联合决策仪表板")
st.caption(f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ================================================================
# 2. 会话状态初始化
# ================================================================

if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'selected_index' not in st.session_state:
    st.session_state.selected_index = None
if 'api' not in st.session_state:
    config = get_config()
    st.session_state.api = PrudenceAPI(config)
if 'data_source_type' not in st.session_state:
    st.session_state.data_source_type = "模拟数据"
if 'custom_data' not in st.session_state:
    st.session_state.custom_data = None
if 'show_comparison' not in st.session_state:
    st.session_state.show_comparison = False
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = []
if 'comparison_customers' not in st.session_state:
    st.session_state.comparison_customers = []
if 'comparison_products' not in st.session_state:
    st.session_state.comparison_products = []


# ================================================================
# 3. 辅助函数：文件解析
# ================================================================

def parse_customers_from_df(df: pd.DataFrame) -> Dict:
    """从DataFrame解析客户数据"""
    customers = {}
    required = ['id', 'risk', 'age', 'assets', 'period']
    for _, row in df.iterrows():
        if all(k in row for k in required):
            cid = str(row['id'])
            customers[cid] = {
                'risk': str(row['risk']),
                'age': int(row['age']),
                'assets': float(row['assets']),
                'period': int(row['period']),
                'first_buy': bool(row.get('first_buy', False)),
                'name': str(row.get('name', '')),
                'income': str(row.get('income', ''))
            }
    return customers


def parse_products_from_df(df: pd.DataFrame) -> Dict:
    """从DataFrame解析产品数据"""
    products = {}
    required = ['id', 'risk', 'name', 'lock', 'min']
    for _, row in df.iterrows():
        if all(k in row for k in required):
            pid = str(row['id'])
            products[pid] = {
                'risk': str(row['risk']),
                'name': str(row['name']),
                'lock': int(row['lock']),
                'min': float(row['min']),
                'type': str(row.get('type', ''))
            }
    return products


def parse_intent_from_df(df: pd.DataFrame) -> Dict:
    """从DataFrame解析意图特征"""
    intent = {}
    for _, row in df.iterrows():
        cid = str(row.get('customer_id', ''))
        if not cid:
            continue
        fname = row.get('feature_name')
        fval = row.get('feature_value')
        if fname is not None and fval is not None:
            if cid not in intent:
                intent[cid] = {}
            intent[cid][fname] = float(fval)
    return intent


def reload_api_with_custom_data(custom_data: Dict):
    """使用自定义数据重新创建API实例"""
    ds = build_dataframe_data_source(
        customers_df=custom_data.get('customers_df'),
        products_df=custom_data.get('products_df'),
        intent_df=custom_data.get('intent_df')
    )
    config = get_config()
    st.session_state.api = PrudenceAPI(config, data_source=ds)
    st.session_state.custom_data = custom_data
    st.session_state.data_source_type = "📂 文件上传"


def reset_to_default_data():
    """重置为默认模拟数据"""
    config = get_config()
    st.session_state.api = PrudenceAPI(config)
    st.session_state.custom_data = None
    st.session_state.data_source_type = "模拟数据"
    st.session_state.show_comparison = False
    st.rerun()


# ================================================================
# 4. 对比展示函数（核心）
# ================================================================

def show_comparison(results: List[Dict], customers: List[str], products: List[str]):
    """展示多人对比结果"""
    st.subheader("📊 多人决策结果对比")

    if not results:
        st.warning("无对比数据")
        return

    # 构建 DataFrame
    df = pd.DataFrame(results)
    df['客户'] = df['customer_id']
    df['产品'] = df['product_id']

    # 提取 Top 信号第一个特征（用于展示）
    df['主要信号'] = df['top_signals'].apply(
        lambda x: x[0]['feature'] if x and len(x) > 0 else '无'
    )

    # ---- 对比表格 ----
    st.subheader("📋 对比明细")
    st.dataframe(
        df[['客户', '产品', 'action', 'suitability_level', 'intent_score', 'rule_score', 'model_score', '主要信号']],
        use_container_width=True,
        column_config={
            'action': '决策动作',
            'suitability_level': '适当性',
            'intent_score': st.column_config.NumberColumn('意图分', format="%.3f"),
            'rule_score': st.column_config.NumberColumn('规则分', format="%.3f"),
            'model_score': st.column_config.NumberColumn('模型分', format="%.3f"),
        }
    )

    # 导出 CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 下载对比数据 (CSV)",
        csv,
        "comparison_results.csv",
        "text/csv",
        key='download-csv'
    )

    # ---- 评分对比柱状图 ----
    st.subheader("📈 各客户-产品意图得分对比")
    fig1 = px.bar(
        df,
        x='客户',
        y='intent_score',
        color='产品',
        barmode='group',
        title='意图得分对比',
        labels={'intent_score': '意图得分', '客户': '客户ID'},
        text_auto='.2f'
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ---- 规则分、模型分、意图分分组对比 ----
    st.subheader("📊 评分维度分解")
    df_melt = df.melt(
        id_vars=['客户', '产品'],
        value_vars=['rule_score', 'model_score', 'intent_score'],
        var_name='评分类型',
        value_name='得分'
    )
    # 映射为中文标签
    score_map = {'rule_score': '规则分', 'model_score': '模型分', 'intent_score': '意图分'}
    df_melt['评分类型'] = df_melt['评分类型'].map(score_map)

    fig2 = px.bar(
        df_melt,
        x='客户',
        y='得分',
        color='评分类型',
        facet_col='产品',
        barmode='group',
        title='各评分维度对比',
        labels={'得分': '分数', '客户': '客户ID'}
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ---- 客户画像雷达图 ----
    st.subheader("🧑‍💼 客户画像雷达图（归一化）")
    profiles = []
    for cid in customers:
        info = st.session_state.api.get_customer(cid)
        # 若获取失败，使用默认值
        if not info:
            info = {"risk": "C3", "age": 40, "assets": 100000, "period": 365}
        profile = {
            '客户': cid,
            '风险等级': int(info.get('risk', 'C1')[1]) if info.get('risk') else 1,
            '年龄': info.get('age', 40),
            '资产(万)': info.get('assets', 100000) / 10000,
            '锁定期(天)': info.get('period', 365),
        }
        profiles.append(profile)
    df_profile = pd.DataFrame(profiles)
    # 归一化处理
    numeric_cols = ['风险等级', '年龄', '资产(万)', '锁定期(天)']
    max_vals = df_profile[numeric_cols].max()
    min_vals = df_profile[numeric_cols].min()
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1
    df_norm = df_profile.copy()
    for col in numeric_cols:
        df_norm[col] = (df_profile[col] - min_vals[col]) / ranges[col]

    fig3 = go.Figure()
    for i, row in df_norm.iterrows():
        fig3.add_trace(go.Scatterpolar(
            r=[row['风险等级'], row['年龄'], row['资产(万)'], row['锁定期(天)']],
            theta=['风险等级', '年龄', '资产(万)', '锁定期(天)'],
            fill='toself',
            name=row['客户']
        ))
    fig3.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="客户画像雷达图"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ---- 适当性等级分布 ----
    st.subheader("📊 适当性等级分布")
    suit_counts = df['suitability_level'].value_counts().reset_index()
    suit_counts.columns = ['等级', '数量']
    fig4 = px.pie(suit_counts, values='数量', names='等级', title='适当性等级分布')
    st.plotly_chart(fig4, use_container_width=True)

    # ---- 决策动作分布 ----
    st.subheader("📊 决策动作分布")
    action_counts = df['action'].value_counts().reset_index()
    action_counts.columns = ['动作', '数量']
    fig5 = px.bar(action_counts, x='动作', y='数量', title='决策动作分布', color='动作')
    st.plotly_chart(fig5, use_container_width=True)

    # ---- Top 信号词云（频次统计） ----
    st.subheader("📌 主要信号出现频次")
    all_signals = []
    for signals in df['top_signals']:
        if signals:
            all_signals.extend([s['feature'] for s in signals[:1]])
    if all_signals:
        signal_series = pd.Series(all_signals).value_counts().reset_index()
        signal_series.columns = ['特征', '频次']
        fig6 = px.bar(signal_series, x='特征', y='频次', title='主要信号出现频次')
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info("无信号数据")

    # ---- 散点图：规则分 vs 模型分 ----
    st.subheader("📈 规则分 vs 模型分 散点图")
    fig7 = px.scatter(
        df,
        x='rule_score',
        y='model_score',
        color='客户',
        size='intent_score',
        hover_data=['产品'],
        title='规则分与模型分关系',
        labels={'rule_score': '规则分', 'model_score': '模型分'}
    )
    st.plotly_chart(fig7, use_container_width=True)


# ================================================================
# 5. 侧边栏
# ================================================================

with st.sidebar:
    st.header("🔧 控制面板")

    # ---- 数据源状态 ----
    st.subheader("📊 数据源状态")
    ds_info = st.session_state.api.get_data_source_info()
    st.info(f"**当前数据源**: {st.session_state.data_source_type}")
    st.caption(f"客户: {ds_info.get('customers_count', 0)} 个 | 产品: {ds_info.get('products_count', 0)} 个")
    st.divider()

    # ---- 文件上传 ----
    st.subheader("📂 加载本地数据文件")
    st.caption("支持 CSV, Excel, JSON 格式")
    uploaded_file = st.file_uploader(
        "选择数据文件",
        type=["csv", "xlsx", "json"],
        help="上传包含客户、产品、意图特征的数据文件"
    )

    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            with st.spinner("正在解析文件..."):
                if file_extension == 'csv':
                    df = pd.read_csv(uploaded_file)
                    if 'customer_id' in df.columns and 'feature_name' in df.columns:
                        custom_data = {'customers_df': None, 'products_df': None, 'intent_df': df}
                    elif 'lock' in df.columns and 'min' in df.columns:
                        custom_data = {'customers_df': None, 'products_df': df, 'intent_df': None}
                    else:
                        custom_data = {'customers_df': df, 'products_df': None, 'intent_df': None}
                elif file_extension == 'xlsx':
                    xlsx = pd.ExcelFile(uploaded_file)
                    sheets = xlsx.sheet_names
                    custom_data = {'customers_df': None, 'products_df': None, 'intent_df': None}
                    for sheet in sheets:
                        df = pd.read_excel(uploaded_file, sheet_name=sheet)
                        sheet_lower = sheet.lower()
                        if 'customer' in sheet_lower or '客户' in sheet_lower:
                            custom_data['customers_df'] = df
                        elif 'product' in sheet_lower or '产品' in sheet_lower:
                            custom_data['products_df'] = df
                        elif 'intent' in sheet_lower or '特征' in sheet_lower or 'feature' in sheet_lower:
                            custom_data['intent_df'] = df
                    if len(sheets) == 1 and all(v is None for v in custom_data.values()):
                        df = pd.read_excel(uploaded_file)
                        if 'customer_id' in df.columns and 'feature_name' in df.columns:
                            custom_data['intent_df'] = df
                        elif 'lock' in df.columns and 'min' in df.columns:
                            custom_data['products_df'] = df
                        else:
                            custom_data['customers_df'] = df
                elif file_extension == 'json':
                    data = json.load(uploaded_file)
                    if isinstance(data, dict):
                        custom_data = {
                            'customers_df': pd.DataFrame(data.get('customers', [])) if data.get('customers') else None,
                            'products_df': pd.DataFrame(data.get('products', [])) if data.get('products') else None,
                            'intent_df': pd.DataFrame(data.get('intent_features', [])) if data.get(
                                'intent_features') else None
                        }
                    else:
                        st.error("JSON格式不正确，请参考示例格式")
                        custom_data = None
                else:
                    st.error("不支持的文件格式")
                    custom_data = None

                if custom_data is not None:
                    reload_api_with_custom_data(custom_data)
                    st.success("✅ 数据加载成功，已更新数据源！")
                    st.rerun()
        except Exception as e:
            st.error(f"❌ 解析文件失败: {e}")

    # ---- 数据源操作按钮 ----
    col_reset, col_demo = st.columns(2)
    with col_reset:
        if st.button("🔄 重置默认", use_container_width=True):
            reset_to_default_data()
            st.rerun()
    with col_demo:
        sample_data = {
            "customers": [
                {"id": "CUST_001", "risk": "C3", "age": 35, "assets": 800000, "period": 365, "first_buy": False,
                 "name": "张三", "income": "高"},
                {"id": "CUST_002", "risk": "C2", "age": 28, "assets": 200000, "period": 90, "first_buy": True,
                 "name": "李四", "income": "中"}
            ],
            "products": [
                {"id": "P001", "risk": "R1", "name": "天天利货币", "lock": 0, "min": 0, "type": "货币型"},
                {"id": "P002", "risk": "R2", "name": "季季盈固收", "lock": 90, "min": 10000, "type": "债券型"}
            ],
            "intent_features": [
                {"customer_id": "CUST_001", "feature_name": "beh_calculator_use_cnt", "feature_value": 3},
                {"customer_id": "CUST_001", "feature_name": "beh_view_cnt_7d", "feature_value": 8}
            ]
        }
        sample_json = json.dumps(sample_data, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 下载示例模板",
            data=sample_json,
            file_name="sample_data.json",
            mime="application/json",
            use_container_width=True
        )

    st.divider()

    # ---- 客户/产品选择 ----
    api = st.session_state.api
    try:
        customers = api.list_customers()
    except Exception:
        customers = ["CUST_HIGH", "CUST_LOW", "CUST_ELDER"]

    try:
        products = api.list_products()
    except Exception:
        products = ["P001", "P002", "P004", "P005", "P006"]

    # ---- 多人对比模式 ----
    st.subheader("👥 多人对比模式")
    enable_comparison = st.checkbox(
        "启用多人对比",
        value=st.session_state.show_comparison,
        help="同时选择多个客户进行决策对比"
    )

    if enable_comparison:
        selected_customers = st.multiselect(
            "选择对比客户",
            options=customers,
            default=st.session_state.comparison_customers or customers[:2] if len(customers) >= 2 else customers,
            help="最多选择 5 个客户"
        )
        if len(selected_customers) > 5:
            st.warning("最多选择 5 个客户")
            selected_customers = selected_customers[:5]

        selected_products = st.multiselect(
            "选择对比产品",
            options=products,
            default=st.session_state.comparison_products or [products[0]] if products else [],
            help="可选择多个产品"
        )
        if not selected_products:
            selected_products = [products[0]] if products else []
    else:
        selected_customers = []
        selected_products = []
        # 单客户模式
        customer_id = st.selectbox(
            "选择客户",
            options=customers,
            index=0 if customers else 0,
            help="从当前数据源加载的客户列表"
        )
        product_id = st.selectbox(
            "选择产品",
            options=products,
            index=min(2, len(products) - 1) if products else 0,
            help="从当前数据源加载的产品列表"
        )

    # ---- 执行决策按钮 ----
    if st.button("🚀 执行决策", type="primary", use_container_width=True):
        with st.spinner("正在执行决策..."):
            try:
                if enable_comparison and selected_customers and selected_products:
                    # 批量对比决策
                    requests = [{"customer_id": c, "product_id": p} for c in selected_customers for p in
                                selected_products]
                    results = api.batch_decide(requests)
                    st.session_state.comparison_results = results
                    st.session_state.comparison_customers = selected_customers
                    st.session_state.comparison_products = selected_products
                    st.session_state.show_comparison = True
                    st.rerun()
                else:
                    # 单客户决策（原有逻辑）
                    if enable_comparison and (not selected_customers or not selected_products):
                        st.error("对比模式下请至少选择 2 个客户和 1 个产品")
                    else:
                        result = api.decide(customer_id, product_id)
                        result['timestamp'] = datetime.now().isoformat()
                        result['customer_id'] = customer_id
                        result['product_id'] = product_id
                        st.session_state.history.append(result)
                        st.session_state.current_result = result
                        st.session_state.selected_index = len(st.session_state.history) - 1
                        st.session_state.show_comparison = False
                        st.rerun()
            except Exception as e:
                st.error(f"❌ 决策失败: {e}")

    st.divider()

    # ---- 历史记录 ----
    st.header("📜 历史记录")
    if st.session_state.history:
        for idx, record in enumerate(st.session_state.history):
            label = f"{record['customer_id']} → {record['product_id']}"
            action = record.get('action', 'UNKNOWN')[:10]
            st.caption(f"{label} ({action}...)")
            if st.button(f"查看 #{idx + 1}", key=f"hist_{idx}"):
                st.session_state.selected_index = idx
                st.session_state.current_result = record
                st.session_state.show_comparison = False
                st.rerun()
        if st.button("🗑️ 清空历史", use_container_width=True):
            st.session_state.history = []
            st.session_state.current_result = None
            st.rerun()
    else:
        st.info("暂无历史记录")

# ================================================================
# 6. 主显示区域
# ================================================================

# 判断显示模式
if st.session_state.get('show_comparison', False):
    # 显示对比结果
    if st.session_state.comparison_results:
        show_comparison(
            st.session_state.comparison_results,
            st.session_state.comparison_customers,
            st.session_state.comparison_products
        )
    else:
        st.info("请先执行决策获得对比数据")
    # 对比模式下不再显示单客户内容
    st.stop()

# ---- 单客户模式显示 ----
if st.session_state.current_result is None and st.session_state.history:
    st.session_state.current_result = st.session_state.history[-1]

if st.session_state.current_result is None:
    st.info("👈 请选择客户和产品，点击「执行决策」或选择历史记录")
    st.stop()

result = st.session_state.current_result
customer_id = result['customer_id']
product_id = result['product_id']

# 获取客户和产品详情
try:
    customer_info = api.get_customer(customer_id)
except Exception:
    customer_info = {"name": "未知客户", "risk": "C3", "age": 40, "assets": 100000, "period": 365}

try:
    product_info = api.get_product(product_id)
except Exception:
    product_info = {"name": "未知产品", "risk": "R3", "lock": 365, "min": 100000}

# ================================================================
# 7. 单客户详细信息展示（原有逻辑）
# ================================================================

col1, col2 = st.columns(2)
with col1:
    st.subheader("📋 客户信息")
    st.write(f"**客户ID**: {customer_id}")
    st.write(f"**姓名**: {customer_info.get('name', 'N/A')}")
    st.write(f"**风险等级**: {customer_info.get('risk', 'N/A')}")
    st.write(f"**年龄**: {customer_info.get('age', 'N/A')} 岁")
    st.write(f"**可投资资产**: ¥{customer_info.get('assets', 0):,}")
    st.write(f"**最长锁定期**: {customer_info.get('period', 'N/A')} 天")

with col2:
    st.subheader("📦 产品信息")
    st.write(f"**产品ID**: {product_id}")
    st.write(f"**产品名称**: {product_info.get('name', 'N/A')}")
    st.write(f"**风险等级**: {product_info.get('risk', 'N/A')}")
    st.write(f"**锁定期**: {product_info.get('lock', 'N/A')} 天")
    st.write(f"**起购金额**: ¥{product_info.get('min', 0):,}")
    st.write(f"**产品类型**: {product_info.get('type', 'N/A')}")

st.divider()
st.subheader("🎯 最终决策")

action = result.get("action", "UNKNOWN")
suit_level = result.get("suitability_level", "UNKNOWN")
intent_score = result.get("intent_score", 0.0)

color_map = {
    "PROACTIVE_CLOSING": "green",
    "NURTURE_CONTENT": "blue",
    "LOW_PRIORITY": "gray",
    "HUMAN_REVIEW_REQUIRED": "orange",
    "BLOCK_AND_REPLACE": "red",
    "ERROR": "red",
}
emoji_map = {
    "PROACTIVE_CLOSING": "🔥",
    "NURTURE_CONTENT": "📚",
    "LOW_PRIORITY": "😴",
    "HUMAN_REVIEW_REQUIRED": "🧑‍💼",
    "BLOCK_AND_REPLACE": "🚫",
    "ERROR": "❌",
}
color = color_map.get(action, "black")
emoji = emoji_map.get(action, "⚖️")

st.markdown(f"## {emoji} 动作：**:{color}[{action}]**")
st.write(f"**适当性等级**: {suit_level}")
st.write(f"**意图得分**: {intent_score:.3f}")
st.write(f"**决策原因**: {result.get('reason', 'N/A')}")
if "latency_ms" in result:
    st.caption(f"⏱️ 决策耗时: {result['latency_ms']}ms | 决策时间: {result.get('timestamp', '')}")

st.divider()
st.subheader("🛡️ 适当性判定详情")

matrix_data = {
    "C1": {"R1": "ALLOW", "R2": "FORBID", "R3": "FORBID", "R4": "FORBID", "R5": "FORBID"},
    "C2": {"R1": "ALLOW", "R2": "ALLOW", "R3": "FORBID", "R4": "FORBID", "R5": "FORBID"},
    "C3": {"R1": "ALLOW", "R2": "ALLOW", "R3": "ALLOW", "R4": "FORBID", "R5": "FORBID"},
    "C4": {"R1": "ALLOW", "R2": "ALLOW", "R3": "ALLOW", "R4": "ALLOW", "R5": "RESTRICTED"},
    "C5": {"R1": "ALLOW", "R2": "ALLOW", "R3": "ALLOW", "R4": "ALLOW", "R5": "ALLOW"},
}

c_risk = customer_info.get("risk", "C3")
p_risk = product_info.get("risk", "R3")
cell_value = matrix_data.get(c_risk, {}).get(p_risk, "UNKNOWN")

st.metric("矩阵匹配结果", cell_value)
st.write(f"**匹配规则**: {c_risk} × {p_risk} → {cell_value}")

replacements = result.get("replacement_products", [])
if replacements:
    st.subheader("🔄 替代推荐产品")
    df_repl = pd.DataFrame(replacements)
    st.dataframe(df_repl[["id", "name", "risk_level", "lock_period", "min_amount"]], use_container_width=True)

st.divider()
st.subheader("📊 意图评分分解")

col3, col4, col5 = st.columns(3)
col3.metric("规则分", f"{result.get('rule_score', 0):.3f}")
col4.metric("模型分", f"{result.get('model_score', 0):.3f}")
col5.metric("融合分", f"{intent_score:.3f}")

st.progress(min(intent_score, 1.0), text=f"意图得分: {intent_score:.1%}")

signals = result.get("top_signals", [])
if signals:
    st.write("**📌 Top 信号特征**")
    for sig in signals:
        impact = sig.get('impact', 'neutral')
        icon = "🟢" if impact == "positive" else ("🔴" if impact == "negative" else "⚪")
        st.write(f"{icon} **{sig.get('feature', 'unknown')}**: 贡献值 {sig.get('shap_value', 0):.3f}")

st.divider()
st.subheader("🗺️ 风险匹配矩阵（全局视图）")

matrix_df = pd.DataFrame(matrix_data).T[["R1", "R2", "R3", "R4", "R5"]]

z_values = []
for row in matrix_df.values:
    z_row = []
    for val in row:
        if val == "ALLOW":
            z_row.append(1)
        elif val == "RESTRICTED":
            z_row.append(2)
        else:
            z_row.append(3)
    z_values.append(z_row)

fig = go.Figure(data=go.Heatmap(
    z=z_values,
    x=matrix_df.columns,
    y=matrix_df.index,
    text=matrix_df.values,
    texttemplate="%{text}",
    colorscale=["green", "orange", "red"],
    showscale=False,
    hoverongaps=False,
))

try:
    row_idx = list(matrix_df.index).index(c_risk)
    col_idx = list(matrix_df.columns).index(p_risk)
    fig.add_trace(go.Scatter(
        x=[matrix_df.columns[col_idx]],
        y=[matrix_df.index[row_idx]],
        mode="markers",
        marker=dict(size=25, color="white", symbol="star", line=dict(width=3, color="black")),
        name="当前选择"
    ))
except Exception:
    pass

fig.update_layout(
    title="⭐ 星标表示当前客户-产品组合",
    height=400,
    width=600,
    xaxis_title="产品风险等级",
    yaxis_title="客户风险等级",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("📄 导出报告")


def generate_html_report(result, customer_info, product_info, matrix_data):
    """生成 HTML 报告"""
    html = f"""
    <html>
    <head><meta charset="UTF-8"><title>睿衡决策报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        .section {{ margin-bottom: 20px; border-bottom: 1px solid #eee; padding-bottom: 15px; }}
        .label {{ font-weight: bold; }}
        .matrix {{ border-collapse: collapse; width: 50%; }}
        .matrix td, .matrix th {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        .allow {{ background-color: #d4edda; }}
        .restricted {{ background-color: #fff3cd; }}
        .forbid {{ background-color: #f8d7da; }}
        .highlight {{ background-color: #cce5ff; }}
    </style>
    </head>
    <body>
    <h1>⚖️ 睿衡引擎 · 决策报告</h1>
    <p>生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>数据源类型：</strong>{st.session_state.data_source_type}</p>

    <div class="section">
        <h2>📋 客户信息</h2>
        <p><span class="label">客户ID：</span>{customer_info.get('id', '')}</p>
        <p><span class="label">风险等级：</span>{customer_info.get('risk', '')}</p>
        <p><span class="label">年龄：</span>{customer_info.get('age', '')}</p>
        <p><span class="label">可投资资产：</span>¥{customer_info.get('assets', 0):,}</p>
    </div>

    <div class="section">
        <h2>📦 产品信息</h2>
        <p><span class="label">产品ID：</span>{product_info.get('id', '')}</p>
        <p><span class="label">产品名称：</span>{product_info.get('name', '')}</p>
        <p><span class="label">风险等级：</span>{product_info.get('risk', '')}</p>
        <p><span class="label">锁定期：</span>{product_info.get('lock', '')}天</p>
    </div>

    <div class="section">
        <h2>🎯 决策结果</h2>
        <p><span class="label">动作：</span><strong>{result.get('action', '')}</strong></p>
        <p><span class="label">适当性等级：</span>{result.get('suitability_level', '')}</p>
        <p><span class="label">意图得分：</span>{result.get('intent_score', 0):.3f}</p>
        <p><span class="label">原因：</span>{result.get('reason', '')}</p>
    </div>

    <div class="section">
        <h2>🛡️ 适当性匹配矩阵</h2>
        <table class="matrix">
            <tr><th>客户\产品</th><th>R1</th><th>R2</th><th>R3</th><th>R4</th><th>R5</th></tr>
    """
    for cr, row in matrix_data.items():
        html += f"<tr><td>{cr}</td>"
        for pr, val in row.items():
            cls = ""
            if cr == customer_info.get('risk') and pr == product_info.get('risk'):
                cls = "highlight"
            elif val == "ALLOW":
                cls = "allow"
            elif val == "RESTRICTED":
                cls = "restricted"
            else:
                cls = "forbid"
            html += f"<td class='{cls}'>{val}</td>"
        html += "</tr>"
    html += """
        </table>
        <p style="color:gray;">⭐ 高亮单元格表示当前选择</p>
    </div>

    <div class="section">
        <h2>📊 意图评分分解</h2>
        <p><span class="label">规则分：</span>{:.3f}</p>
        <p><span class="label">模型分：</span>{:.3f}</p>
        <p><span class="label">融合分：</span>{:.3f}</p>
    </div>
    """.format(
        result.get('rule_score', 0),
        result.get('model_score', 0),
        result.get('intent_score', 0)
    )

    replacements = result.get("replacement_products", [])
    if replacements:
        html += '<div class="section"><h2>🔄 替代推荐产品</h2><ul>'
        for r in replacements:
            html += f"<li>{r.get('name', '')} (R{r.get('risk_level', '')}) - 锁定期{r.get('lock_period', 0)}天</li>"
        html += "</ul></div>"

    html += "</body></html>"
    return html


customer_info_with_id = {**customer_info, "id": customer_id}
product_info_with_id = {**product_info, "id": product_id}
report_html = generate_html_report(result, customer_info_with_id, product_info_with_id, matrix_data)

st.download_button(
    label="📥 下载 HTML 报告",
    data=report_html,
    file_name=f"睿衡决策报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
    mime="text/html",
    use_container_width=True,
)
st.caption("💡 提示：HTML 报告可直接在浏览器中打印为 PDF")

st.divider()
st.caption("⚖️ 睿衡引擎 v2.0 | 数据源: {} | 如有问题请联系管理员".format(st.session_state.data_source_type))