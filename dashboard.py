"""One linked customer dashboard: scatter -> profile -> economics -> service plan."""
from dataclasses import asdict
import hashlib
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import optimize_allocation
from config import get_config
from customer_scoring import (DIMENSIONS, SIGNALS, ValueScenario, composite, customer_scores,
                              factor_inputs, fit_factors, scenario_values, selected_customer_ids)
from dashboard_data import (activate_source, chart, csv_bytes, data_page, download_csv, hero, style)
from main import PrudenceAPI
from workbench_data import demo_source, source_tables

LEVELS = {"ALLOW": "可通过", "RESTRICTED": "需复核", "FORBID": "已拦截"}
COLORS = {"可通过": "#168577", "需复核": "#d39436", "已拦截": "#c96970"}
SYMBOLS = {"可通过": "circle", "需复核": "diamond", "已拦截": "x"}
MODES = ["关注度 × 匹配度", "关注度 × 年度贡献", "因子 1 × 因子 2"]
DISPLAY = {"customer_id": "客户 ID", "name": "客户", "score": "综合观察分", "engagement": "关注度",
           "fit": "产品匹配度", "status": "适当性", "annual_contribution": "情景年度贡献（元）",
           "coverage": "画像完整维度", "missing": "待补充指标"}


@st.cache_data(show_spinner=False)
def factors_for(frame, n_factors):
    return fit_factors(frame, n_factors)


def scenario_controls(product_id, original):
    saved = st.session_state.get("scenario_settings", {})
    scenario = ValueScenario(**saved.get("rates", {}))
    product = saved.get("products", {}).get(product_id, original).copy()
    if st.session_state.get("scenario_form_product") != product_id:
        prefixes = [f"rate_{name}" for name in ("deposit_rate", "loan_rate", "funding_rate", "wealth_return", "wealth_fee")]
        for prefix in prefixes + ["ds", "ws", "loan", "pd", "lgd", "cost", "risk", "lock", "min"]:
            st.session_state.pop(f"{prefix}_{product_id}", None)
        st.session_state.scenario_form_product = product_id
    with st.sidebar.expander("利率与理财策略", expanded=False):
        st.caption("以下为自定义年化情景假设，不是实时银行报价。提交后全看板联动。")
        with st.form(f"scenario_form_{product_id}"):
            rates = {}
            for key, label in [("deposit_rate", "存款利率（%）"), ("loan_rate", "贷款利率（%）"),
                               ("funding_rate", "内部资金转移利率 FTP（%）"),
                               ("wealth_return", "理财假设毛收益率（%）"), ("wealth_fee", "理财年费率（%）")]:
                rates[key] = st.number_input(label, -100. if key == "wealth_return" else 0., 100., getattr(scenario, key) * 100,
                                            .1, key=f"rate_{key}_{product_id}") / 100
            rates["deposit_share"] = st.slider("存款资金占比（%）", 0, 100, round(scenario.deposit_share * 100), key=f"ds_{product_id}") / 100
            rates["wealth_share"] = st.slider("理财资金占比（%）", 0, 100, round(scenario.wealth_share * 100), key=f"ws_{product_id}") / 100
            rates["loan_assumption"] = st.number_input("缺少贷款余额时的假设（万元 / 人）", 0., 100000., scenario.loan_assumption / 10000, key=f"loan_{product_id}") * 10000
            rates["default_probability"] = st.number_input("贷款年度违约概率假设（%）", 0., 100., scenario.default_probability * 100, key=f"pd_{product_id}") / 100
            rates["loss_given_default"] = st.number_input("违约损失率假设（%）", 0., 100., scenario.loss_given_default * 100, key=f"lgd_{product_id}") / 100
            rates["annual_cost"] = st.number_input("年度服务成本（元 / 人）", 0., 1000000., scenario.annual_cost, key=f"cost_{product_id}")
            st.markdown("**当前理财产品条件**")
            risk = st.selectbox("产品风险等级", [f"R{i}" for i in range(1, 6)], index=int(product["risk"][1:]) - 1, key=f"risk_{product_id}")
            lock = st.number_input("锁定期限（天）", 0, 36500, int(product["lock"]), key=f"lock_{product_id}")
            minimum = st.number_input("最低金额（元）", 0., 1e10, float(product["min"]), key=f"min_{product_id}")
            if st.form_submit_button("应用利率与策略", type="primary"):
                try:
                    proposed = ValueScenario(**rates)
                    proposed.validate()
                    products = saved.get("products", {}).copy()
                    products[product_id] = dict(original, risk=risk, lock=lock, min=minimum)
                    st.session_state.scenario_settings = dict(rates=asdict(proposed), products=products)
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))
        st.caption("仅作用于本次看板；数据源中的产品与 API 规则不被改写。存款与理财共享可用资产池，合计不能超过 100%。")
    return scenario, product


def scatter_figure(frame, x, y, mode):
    fig = px.scatter(frame, x=x, y=y, color="status", symbol="status",
        color_discrete_map=COLORS, symbol_map=SYMBOLS, custom_data=["customer_id", "name", "score", "status"],
        category_orders={"status": list(COLORS)})
    fig.update_traces(marker=dict(size=11, opacity=.8, line=dict(width=.7, color="white")),
        hovertemplate="<b>%{customdata[1]}</b> · %{customdata[0]}<br>横轴 %{x:.1f} · 纵轴 %{y:.1f}<br>综合观察分 %{customdata[2]:.1f}<br>%{customdata[3]}<extra></extra>")
    if mode != MODES[2]:
        fig.update_xaxes(range=[-4, 104], title="关注度 / 100")
        fig.add_vline(x=60, line_dash="dot", line_color="#c2d1d6")
        if mode == MODES[0]:
            fig.update_yaxes(range=[-4, 106], title="产品匹配度 / 100")
            fig.add_hline(y=60, line_dash="dot", line_color="#c2d1d6")
            for x0, y0, label in [(0, 102, "匹配较高 · 待了解"), (100, 102, "关注与匹配均较高"),
                                  (0, 3, "补充需求信息"), (100, 3, "关注较高 · 检查适配")]:
                fig.add_annotation(x=x0, y=y0, text=label, showarrow=False, xanchor="left" if x0 == 0 else "right",
                                   font=dict(size=10, color="#7e929b"))
        else:
            fig.update_yaxes(title="情景年度贡献 / 元")
            fig.add_hline(y=0, line_dash="dot", line_color="#c2d1d6")
    else:
        fig.add_hline(y=0, line_dash="dot", line_color="#d9e2e6")
        fig.add_vline(x=0, line_dash="dot", line_color="#d9e2e6")
    fig.update_layout(template="plotly_white", height=490, margin=dict(l=20, r=20, t=15, b=50),
        paper_bgcolor="white", plot_bgcolor="white", clickmode="event+select", dragmode="pan",
        legend=dict(orientation="h", y=-.15, title_text=""), font=dict(color="#45616a"))
    fig.update_xaxes(gridcolor="#edf2f3", zeroline=False)
    fig.update_yaxes(gridcolor="#edf2f3", zeroline=False)
    return fig


def radar_figure(row, cohort):
    labels = DIMENSIONS + [DIMENSIONS[0]]
    fig = go.Figure()
    median = cohort[DIMENSIONS].median()
    fig.add_trace(go.Scatterpolar(r=median.tolist() + [median.iloc[0]], theta=labels, name="筛选群体中位数",
                                 line=dict(color="#9faeb7", dash="dot"), hovertemplate="%{theta} %{r:.1f}<extra></extra>"))
    values = row[DIMENSIONS].tolist()
    fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=labels, name="当前客户",
        fill="toself" if np.isfinite(values).all() else None, fillcolor="rgba(22,133,119,.15)",
        line=dict(color="#168577", width=2.5), mode="lines+markers", connectgaps=False,
        hovertemplate="%{theta} %{r:.1f}<extra></extra>"))
    fig.update_layout(template="plotly_white", height=340, margin=dict(l=58, r=58, t=25, b=40),
        polar=dict(radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75, 100], tickfont=dict(size=9)),
                   angularaxis=dict(tickfont=dict(size=11)), bgcolor="white"),
        legend=dict(orientation="h", y=-.2, x=.1), font=dict(color="#45616a"), paper_bgcolor="white")
    return fig


def manual_focus_changed():
    st.session_state.selection_nonce = st.session_state.get("selection_nonce", 0) + 1


def profile_panel(frame, scenario):
    ids = frame.customer_id.tolist()
    cid = st.selectbox("当前客户画像", ids, key="focus_customer", on_change=manual_focus_changed,
                       format_func=lambda c: f"{frame.loc[c].get('name', c)} · {c}")
    row = frame.loc[cid]
    st.caption(f"{row.status} · {row['risk']} · {row['age']} 岁 · 可用资产 ¥{row.assets:,.0f}")
    st.plotly_chart(radar_figure(row, frame), use_container_width=True, key="customer_radar", config={"displaylogo": False})
    a, b = st.columns(2)
    a.metric("综合观察分", f"{row.score:.1f}" if pd.notna(row.score) else "待补数据")
    b.metric("情景年度贡献", f"¥{row.annual_contribution:,.0f}")
    if row.suitability_level == "FORBID":
        st.error("当前理财产品已拦截；不进入服务分配。" + str(row.reason))
    elif row.suitability_level == "RESTRICTED":
        st.warning("当前理财产品需人工复核；不进入服务分配。" + str(row.reason))
    else:
        st.caption("当前规则可通过。下一步：核实客户需求、资金安排与服务意愿。")
    if row.missing:
        st.caption("待补充：" + row.missing + "。雷达图留空，不记作零分。")
    return row


def profile_details(row, scenario):
    with st.expander("所选客户 · 评分依据与贡献拆解", expanded=False):
        a, b = st.columns([1, 1.3])
        with a:
            st.dataframe(pd.DataFrame({"维度": DIMENSIONS, "得分 / 100": row[DIMENSIONS].values}), hide_index=True, use_container_width=True)
            st.dataframe(pd.DataFrame({"原始行为指标": list(SIGNALS.values()), "观测值": [row.signals[k] for k in SIGNALS]}), hide_index=True, use_container_width=True)
        with b:
            values = [row.deposit_contribution, row.loan_contribution, row.wealth_contribution,
                      -row.expected_loss, -scenario.annual_cost, row.annual_contribution]
            chart(go.Figure(go.Waterfall(x=["存款利差", "贷款利差", "理财费用", "预期损失", "服务成本", "年度贡献"],
                y=values, measure=["relative"] * 5 + ["total"], text=[f"¥{v:,.0f}" for v in values],
                textposition="outside", increasing=dict(marker=dict(color="#168577")),
                decreasing=dict(marker=dict(color="#c96970")), totals=dict(marker=dict(color="#355a70")))).update_layout(title="同一客户的年度贡献情景"))
            st.caption(f"存款配置 ¥{row.deposit_balance:,.0f} · 理财配置 ¥{row.wealth_balance:,.0f} · 未配置 ¥{row.unallocated:,.0f}。")
            st.caption(f"贷款余额 ¥{row.loan_balance:,.0f}（{'情景假设' if row.loan_is_assumed else '导入记录'}）；客户年度利息与理财净收支情景 ¥{row.customer_annual_cashflow:,.0f}。")
            st.caption("客户收支 = 存款利息 + 理财假设毛收益 − 理财费用 − 贷款利息；不含本金、税费、净值波动。年度贡献为机构口径，非客户收益。")


def structural_analysis(factors, factor_error, frame, row, alpha):
    st.caption("因子模型始终使用当前数据源全体完整客户，筛选不会重新拟合；主图切换到因子视图仍可点击查看同一客户。")
    if factors is None:
        st.info(factor_error)
    else:
        a, b, c = st.columns(3)
        a.metric("因子分析有效样本", factors["n"])
        b.metric("协方差重建相对误差", f"{factors['residual']:.1%}")
        c.metric("相关矩阵条件数", f"{factors['condition']:.1f}")
        left, right = st.columns([1.3, 1])
        with left:
            chart(px.imshow(factors["loadings"], text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="BrBG", title="旋转因子载荷 · 特征如何共同变化"), 390)
        with right:
            for name in factors["loadings"]:
                strongest = factors["loadings"][name].abs().nlargest(2).index
                st.write(f"**{name}** · 主要关联：{'、'.join(strongest)}")
                if row.customer_id in factors["scores"].index:
                    st.caption(f"当前客户因子得分：{factors['scores'].loc[row.customer_id, name]:.2f}")
            st.dataframe(pd.DataFrame({"共同度": factors["communalities"], "独特方差": factors["uniqueness"]}).round(3), use_container_width=True)
        st.caption(f"完整案例删除 {factors['dropped_rows']} 位；常量列排除：{'、'.join(factors['dropped_columns']) or '无'}。采用标准化、最大似然因子分析与 Varimax 旋转。因子符号和次序不代表优劣，也不证明因果或预测能力。")
        if factors["condition"] > 1000:
            st.warning("变量高度共线，因子结构可能不稳定，应补充样本并复核变量选择。")
    st.markdown("**权重敏感性 · 所选客户在当前群体中的名次**")
    sensitivity = []
    for weight in np.arange(.1, 1., .1):
        scores = composite(frame.engagement, frame.fit, weight)
        ranks = scores.rank(method="min", ascending=False)
        sensitivity.append({"关注度权重": round(weight * 100), "名次": ranks.loc[row.customer_id], "观察分": scores.loc[row.customer_id]})
    sensitivity = pd.DataFrame(sensitivity)
    if sensitivity["名次"].notna().any():
        fig = px.line(sensitivity, x="关注度权重", y="名次", markers=True)
        fig.update_yaxes(autorange="reversed", dtick=1 if len(frame) < 15 else None)
        fig.add_vline(x=alpha * 100, line_dash="dot", line_color="#168577")
        chart(fig, 250)
        st.caption("其余条件固定，观察分 = 100 × (关注度 / 100)^权重 × (匹配度 / 100)^(1−权重)。并列使用最小名次；名次随筛选群体变化。")
    else:
        st.info("所选客户数据不完整，暂不计算名次敏感性。")


def service_plan(frame, product_id, selection, scenario_context):
    st.caption("把同一看板中的客户转为本次服务候选。仅纳入数据完整、适当性可通过、有有效理财配置且目标效用为正的客户。此处只生成计划。")
    a, b, c, d = st.columns(4)
    target = a.selectbox("分配目标", ["综合观察分", "情景年度贡献"], key="allocation_target")
    budget = b.number_input("本次触达预算（元）", 0., 1e8, 1000., key="allocation_budget")
    cost = c.number_input("每次触达成本（元）", 1., 1e6, 50., key="contact_cost")
    limit = d.number_input("本次最多触达人数", 0, 2000, 10, key="contact_limit")
    only = st.checkbox(f"只使用图中圈选客户（{len(selection)} 位）", value=False, key="selected_only")
    pool = frame.loc[frame.customer_id.isin(selection)] if only else frame
    utility = "score" if target == "综合观察分" else "annual_contribution"
    pool = pool.loc[pool.score.notna() & pool.suitability_level.eq("ALLOW") &
                    pool.wealth_balance.gt(0) & pool[utility].gt(0)].copy()
    candidates = pd.DataFrame({"customer_id": pool.customer_id, "product_id": product_id,
        "suitability_level": pool.suitability_level, "utility": pool[utility], "cost": cost})
    signature = hashlib.sha256((candidates.to_json() + str((budget, cost, limit, target)) + scenario_context).encode()).hexdigest()
    if st.button("生成服务计划", type="primary"):
        try:
            result = optimize_allocation(candidates, budget, int(limit), {product_id: int(limit)})
            st.session_state.allocation = (signature, result)
        except ValueError as error:
            st.error(str(error))
    previous = st.session_state.get("allocation")
    if previous and previous[0] != signature:
        st.info("候选或参数已改变，请重新生成服务计划。")
    elif previous:
        result = previous[1]
        st.success(f"计划服务 {len(result.selected)} 位 · 本次成本 ¥{result.cost:,.0f} · 目标效用合计 {result.objective:,.1f}")
        chosen = pool.loc[pool.customer_id.isin(result.selected.customer_id)]
        output = chosen[list(DISPLAY)].rename(columns=DISPLAY)
        st.dataframe(output, hide_index=True, use_container_width=True)
        download_csv(output, "下载服务计划", "customer_service_plan.csv")
    st.caption("采用 0–1 整数规划，每位客户最多一次。本次触达成本与年度服务成本是独立假设；情景年度贡献是存量规模指标，不能解释为此次触达带来的增量利润。")


def methodology(scenario, product, alpha):
    st.markdown("**分数回答『有哪些观察信号』，价值情景回答『在这些假设下贡献多少』。**")
    st.markdown("六维均为 0–100：活跃度组合浏览次数与购买间隔；投入组合计算器与比较次数；响应组合打开率、顾问联系与话术接受；另三维为当前产品的风险、资金和期限匹配。具体阈值是本项目的可解释工程约定，尚未经真实转化标签校准。")
    st.markdown("关注度为前三维均值，匹配度为后三维均值。综合观察分采用加权几何平均，任一维缺失则不生成综合分。散点参考线 60 是阅读辅助线，不是统计显著阈值。风险适配序数比值只作展示，最终适当性始终由原有硬规则决定。")
    st.markdown("**年度贡献（元）** = 存款金额 × (FTP − 存款利率) + 贷款余额 × (贷款利率 − FTP) + 理财配置 × 年费率 − 贷款余额 × 违约概率 × 违约损失率 − 年度服务成本。")
    st.markdown("金额按固定余额、持有一整年计算；不含税、资本占用、期限匹配的流动性溢价、违约后利息修正和未来留存。理财收益率影响客户收支，理财费率影响机构费用收入；二者分别计算。适当性金额规则按起投额与计划理财金额的较大者重新检查。被拦截、待复核或未达起投金额的理财配置为零，资金保留为未配置，不自动转投。")
    st.markdown("**研究与实现参考**")
    st.markdown("- [llm_benchmark](https://github.com/llm2014/llm_benchmark)：学习同一筛选集驱动散点与榜单、轴含义清晰的组织方式；客户画像与评分在本项目独立实现。\n- [Shneiderman, 1996 · The Eyes Have It](https://hci.stanford.edu/courses/cs448b/papers/shneiderman96eyes.pdf)：全局总览、筛选及按需查看明细，落地为点选客户雷达。\n- [Aliyev 等, 2020 · 银行客户 RFM 分群](https://arxiv.org/abs/2008.08662)：借鉴行为分层问题；本项目无完整交易流水，因此不冒称实现标准 RFM 或生命周期价值。\n- [OECD / JRC, 2008 · 复合指标手册](https://doi.org/10.1787/9789264043466-en)：明确归一化、权重和敏感性；本项目的数值锚点不来自该手册。\n- [scikit-learn · 旋转因子分析](https://scikit-learn.org/stable/auto_examples/decomposition/plot_varimax_fa.html)：以载荷和独特方差解释共同结构，避免把因子得分当价值标签。")
    st.download_button("下载当前情景参数", json.dumps(dict(scenario=asdict(scenario), product=product,
        attention_weight=alpha, score_version="customer-observation-v1"), ensure_ascii=False, indent=2),
        "customer_scenario.json", "application/json")


def main():
    st.set_page_config(page_title="睿衡 · 客户价值看板", page_icon="◈", layout="wide")
    style()
    if "api" not in st.session_state:
        with st.spinner("正在准备客户数据…"):
            config = get_config()
            if config.data_source.type == "mock":
                activate_source(demo_source(), "模拟数据 · 120 位客户 · seed 42")
            else:
                st.session_state.api = PrudenceAPI(config)
                st.session_state.source_label = f"配置数据源 · {config.data_source.type}"
    api = st.session_state.api
    customers, products = source_tables(api.data_source)
    if customers.empty or products.empty:
        st.info("请先导入非空客户表和产品表。")
        data_page(api, customers, products)
        return
    with st.sidebar:
        st.markdown('<div class="brand">◈ 睿衡引擎</div><div class="brand-sub">CUSTOMER INTELLIGENCE</div>', unsafe_allow_html=True)
        st.caption(st.session_state.source_label)
        ids = products.id.tolist()
        product_id = st.selectbox("分析理财产品", ids, index=ids.index("P004") if "P004" in ids else 0,
            key="product_id", format_func=lambda pid: f"{products.set_index('id').loc[pid, 'name']} · {pid}")
    original = api.data_source.get_product(product_id)
    scenario, product = scenario_controls(product_id, original)
    with st.sidebar:
        st.caption(f"生效条件：{product['risk']} · {product['lock']} 天 · ¥{product['min']:,.0f} 起")
        with st.expander("评分与因子设置"):
            alpha = st.slider("关注度权重（%）", 10, 90, 50, 10, key="attention_weight") / 100
            n_factors = st.selectbox("提取因子数", [2, 3], key="n_factors")
        st.divider()
        levels = st.multiselect("适当性筛选", list(LEVELS.values()), default=list(LEVELS.values()), key="level_filter")
        search = st.text_input("搜索客户姓名或 ID", key="customer_search")
        st.caption("筛选只改变观察人群，不改变同一客户的评分、金额假设或因子坐标。")
    frame = scenario_values(customer_scores(api.data_source, api.engines["suitability"], product_id,
        product, alpha, planned_wealth_share=scenario.wealth_share), product, scenario)
    frame["status"] = frame.suitability_level.map(LEVELS)
    factors, factor_error = None, ""
    try:
        factors = factors_for(factor_inputs(api.data_source), n_factors)
        frame = frame.join(factors["scores"])
    except ValueError as error:
        factor_error = str(error)
    visible = frame.loc[frame.status.isin(levels)].copy()
    if search:
        visible = visible.loc[visible.customer_id.str.contains(search, case=False, regex=False) |
                              visible['name'].fillna('').str.contains(search, case=False, regex=False)]
    hero("PRUDENCE / CUSTOMER INTELLIGENCE", "看见客户特征，理解价值来源。", "一位客户，一个点。点击散点查看六维画像；调整利率和理财策略，探索同一群体的价值变化。")
    st.caption(f"{st.session_state.source_label}　/　产品：{product.get('name', product_id)}　/　观察评分与年度价值情景")
    a, b, c, d = st.columns(4)
    a.metric("当前客户", len(visible), help="受左侧客户筛选影响")
    b.metric("完整画像", f"{visible.score.notna().sum()} / {len(visible)}")
    c.metric("规则可通过", int(visible.suitability_level.eq("ALLOW").sum()))
    d.metric("情景年度贡献合计", f"¥{visible.annual_contribution.sum():,.0f}", help="含假设余额、利率、损失与成本；不是实际利润或客户终身价值。")
    mode = st.radio("客户分布视角", MODES, horizontal=True, key="scatter_mode")
    if not visible.empty:
        if st.session_state.get("focus_customer") not in visible.index:
            st.session_state.focus_customer = visible.score.sort_values(ascending=False, na_position="last").index[0]
        x, y = ("engagement", "fit") if mode == MODES[0] else ("engagement", "annual_contribution")
        if mode == MODES[2]:
            if factors is not None:
                x, y = "因子 1", "因子 2"
            else:
                st.info(factor_error + " 当前先显示关注度与匹配度。")
                mode, x, y = MODES[0], "engagement", "fit"
        plotted = visible.dropna(subset=[x, y])
        context = hashlib.sha256((plotted[["customer_id", x, y, "status"]].to_json() + mode + str(alpha) +
                                  product_id + json.dumps(asdict(scenario), sort_keys=True) +
                                  str(st.session_state.get("selection_nonce", 0))).encode()).hexdigest()[:16]
        key = f"customer_scatter_v1_{context}"
        st.session_state.scatter_key = key
        left, right = st.columns([1.65, 1], gap="large")
        with left:
            st.markdown("**客户分布**")
            st.caption("点选查看画像 · 框选 / 套索形成服务候选 · 同位置客户可从右侧名单选择")
            if not plotted.empty:
                event = st.plotly_chart(scatter_figure(plotted, x, y, mode),
                    use_container_width=True, key=key, on_select="rerun", selection_mode=("points", "box", "lasso"), config={"displaylogo": False})
                selection = selected_customer_ids(event, plotted.index)
            else:
                st.info("当前视角没有完整坐标。仍可从右侧选择客户查看已有画像。")
                selection = []
            signature = (context, tuple(selection))
            if st.session_state.get("last_point_selection") != signature:
                if selection:
                    st.session_state.focus_customer = selection[0]
                st.session_state.last_point_selection = signature
                st.session_state.selected_customers = selection
            st.caption(f"显示 {len(plotted)} 个点 · {len(visible) - len(plotted)} 位缺少当前坐标 · 已圈选 {len(selection)} 位。颜色与点形表示适当性，分数不能覆盖拦截结果。")
        with right:
            row = profile_panel(visible, scenario)
        profile_details(row, scenario)
        tabs = st.tabs(["客户榜单", "共同因子与稳定性", "服务计划", "口径与研究", "数据与导入"])
        with tabs[0]:
            ranked = visible.sort_values("score", ascending=False, na_position="last")
            output = ranked[list(DISPLAY)].rename(columns=DISPLAY)
            st.dataframe(output.round(2), hide_index=True, use_container_width=True)
            download_csv(output, "下载当前客户画像与价值", "customer_value_board.csv")
        with tabs[1]:
            structural_analysis(factors, factor_error, visible, row, alpha)
        with tabs[2]:
            service_plan(visible, product_id, st.session_state.get("selected_customers", []),
                         json.dumps(dict(scenario=asdict(scenario), product=product, alpha=alpha), sort_keys=True))
        with tabs[3]:
            methodology(scenario, product, alpha)
        with tabs[4]:
            data_page(api, customers, products)
    else:
        st.info("当前筛选没有客户，请调整条件。")
        with st.expander("数据与导入"):
            data_page(api, customers, products)
    st.markdown('<div class="footer">PRUDENCE ENGINE · 客户价值看板　/　数据有来源，评分有依据，情景可复算。</div>', unsafe_allow_html=True)
