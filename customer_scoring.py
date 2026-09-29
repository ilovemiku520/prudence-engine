"""Transparent customer observation scores, scenario economics and latent factors.

Scores are engineering conventions, not calibrated purchase probabilities or credit ratings.
The scenario is annual, static and pre-tax; no lifetime-value claim is made.
"""
from dataclasses import asdict, dataclass
import warnings

import numpy as np
import pandas as pd

DIMENSIONS = ["近期活跃", "决策投入", "互动响应", "风险适配", "资金余量", "期限匹配"]
SIGNALS = {
    "beh_view_cnt_7d": "近 7 日浏览次数",
    "txn_days_since_last_purchase": "距上次购买天数",
    "beh_calculator_use_cnt": "计算器使用次数",
    "beh_compare_cnt": "产品比较次数",
    "interact_push_open_rate_30d": "近 30 日推送打开率",
    "interact_advisor_contact_freq": "顾问联系频次",
    "interact_last_script_accepted": "最近话术接受标记",
}


def observed(value, upper=None):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return np.nan
    return value if np.isfinite(value) and value >= 0 and (upper is None or value <= upper) else np.nan


def saturation(value, anchor):
    return 100 * np.clip(np.log1p(value) / np.log1p(anchor), 0, 1)


def composite(engagement, fit, attention_weight=.5):
    if not np.isfinite(attention_weight) or not 0 < attention_weight < 1:
        raise ValueError("关注度权重必须在 0 与 1 之间。")
    return 100 * (engagement / 100) ** attention_weight * (fit / 100) ** (1 - attention_weight)


def customer_scores(source, engine, product_id, product, attention_weight=.5, planned_wealth_share=0.):
    if not np.isfinite(planned_wealth_share) or not 0 <= planned_wealth_share <= 1:
        raise ValueError("理财资金占比必须在 0–100% 之间。")
    rows = []
    for cid in source.list_customers():
        customer = source.get_customer(cid)
        features = source.get_intent_features(cid, product_id)
        f = {k: observed(features.get(k), 1 if k in {
            "interact_push_open_rate_30d", "interact_last_script_accepted"} else None) for k in SIGNALS}
        # Arithmetic propagates missing values: absence is never evidence of zero activity.
        activity = (saturation(f["beh_view_cnt_7d"], 20) + 100 * 2 ** (-f["txn_days_since_last_purchase"] / 30)) / 2
        effort = (saturation(f["beh_calculator_use_cnt"], 5) + saturation(f["beh_compare_cnt"], 10)) / 2
        response = (50 * f["interact_push_open_rate_30d"] +
                    .25 * saturation(f["interact_advisor_contact_freq"], 5) +
                    25 * f["interact_last_script_accepted"])
        assets, period = observed(customer.get("assets")), observed(customer.get("period"))
        risk = 100 * min(int(customer["risk"][1:]) / int(product["risk"][1:]), 1)
        funds = 100 * max(0, 1 - product["min"] / assets) if assets > 0 else 0 if assets == 0 else np.nan
        horizon = 100 * min(period / product["lock"], 1) if product["lock"] > 0 else 100.
        dims = [activity, effort, response, risk, funds, horizon]
        engagement, fit = np.mean(dims[:3]), np.mean(dims[3:])
        # Apply the existing financial rules to the actual proposed amount too.
        checked_amount = max(product["min"], assets * planned_wealth_share)
        result = engine.check_suitability(
            dict(id=cid, risk_level=customer["risk"], age=customer["age"], available_assets=assets,
                 max_acceptable_period=period, first_time_buyer=customer.get("first_buy", False)),
            dict(id=product_id, risk_level=product["risk"], lock_period=product["lock"], min_amount=checked_amount))
        rows.append(dict(customer, customer_id=cid, name=customer.get("name") or cid, **dict(zip(DIMENSIONS, dims)),
                         engagement=engagement, fit=fit, score=composite(engagement, fit, attention_weight),
                         coverage=sum(np.isfinite(dims)), suitability_level=result.level,
                         reason=result.restriction_reason or result.matched_rule,
                         signals=f, missing="、".join(SIGNALS[k] for k, v in f.items() if not np.isfinite(v))))
    return pd.DataFrame(rows).set_index("customer_id", drop=False)


@dataclass(frozen=True)
class ValueScenario:
    deposit_rate: float = .015
    loan_rate: float = .04
    funding_rate: float = .025
    wealth_return: float = .035
    wealth_fee: float = .005
    deposit_share: float = .6
    wealth_share: float = .4
    loan_assumption: float = 0.
    default_probability: float = .01
    loss_given_default: float = .45
    annual_cost: float = 100.

    def validate(self):
        if not all(np.isfinite(v) and (v >= -1 if k == "wealth_return" else v >= 0)
                   for k, v in asdict(self).items()):
            raise ValueError("情景参数必须为有限数值；理财收益率最低 −100%，其他参数不能为负。")
        for name in ("deposit_rate", "loan_rate", "funding_rate", "wealth_return", "wealth_fee",
                     "deposit_share", "wealth_share", "default_probability", "loss_given_default"):
            if getattr(self, name) > 1:
                raise ValueError("利率、费率与占比必须在 0–100% 之间。")
        if self.deposit_share + self.wealth_share > 1 + 1e-10:
            raise ValueError("存款与理财资金占比之和不能超过 100%。")


def scenario_values(frame, product, scenario):
    scenario.validate()
    result = frame.copy()
    assets = pd.to_numeric(result.assets, errors="coerce")
    if not np.isfinite(assets).all() or (assets < 0).any():
        raise ValueError("可用资产必须为非负有限数值。")
    result["deposit_balance"] = assets * scenario.deposit_share
    proposed = assets * scenario.wealth_share
    allowed = result.suitability_level.eq("ALLOW") & (proposed >= product["min"])
    result["wealth_balance"] = proposed.where(allowed, 0.)
    result["unallocated"] = assets - result.deposit_balance - result.wealth_balance
    loan = result.get("loan_balance", pd.Series(np.nan, index=result.index)).map(observed)
    result["loan_is_assumed"] = loan.isna()
    result["loan_balance"] = loan.fillna(scenario.loan_assumption)
    result["deposit_contribution"] = result.deposit_balance * (scenario.funding_rate - scenario.deposit_rate)
    result["loan_contribution"] = result.loan_balance * (scenario.loan_rate - scenario.funding_rate)
    result["wealth_contribution"] = result.wealth_balance * scenario.wealth_fee
    result["expected_loss"] = result.loan_balance * scenario.default_probability * scenario.loss_given_default
    result["annual_contribution"] = (result.deposit_contribution + result.loan_contribution +
                                     result.wealth_contribution - result.expected_loss - scenario.annual_cost)
    result["customer_annual_cashflow"] = (result.deposit_balance * scenario.deposit_rate +
        result.wealth_balance * (scenario.wealth_return - scenario.wealth_fee) - result.loan_balance * scenario.loan_rate)
    return result


def selected_customer_ids(event, valid_ids):
    """Use stable IDs in customdata, never trace-local point indices."""
    if not isinstance(event, dict):
        return []
    selection = event.get("selection", {})
    if not isinstance(selection, dict):
        return []
    valid = set(valid_ids)
    found = []
    for point in selection.get("points", []):
        if not isinstance(point, dict):
            continue
        data = point.get("customdata", [])
        cid = data[0] if isinstance(data, (list, tuple)) and data else None
        if isinstance(cid, str) and cid in valid and cid not in found:
            found.append(cid)
    return found


def factor_inputs(source):
    rows = {}
    for cid in source.list_customers():
        f = source.get_intent_features(cid)
        rows[cid] = {
            "可用资产·log": np.log1p(observed(source.get_customer(cid).get("assets"))),
            "浏览次数·log": np.log1p(observed(f.get("beh_view_cnt_7d"))),
            "计算器使用·log": np.log1p(observed(f.get("beh_calculator_use_cnt"))),
            "产品比较·log": np.log1p(observed(f.get("beh_compare_cnt"))),
            "购买间隔·log": np.log1p(observed(f.get("txn_days_since_last_purchase"))),
            "推送打开率": observed(f.get("interact_push_open_rate_30d"), 1),
            "顾问联系·log": np.log1p(observed(f.get("interact_advisor_contact_freq"))),
            "持有期限·log": np.log1p(observed(f.get("txn_avg_holding_period"))),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def fit_factors(frame, n_factors=2):
    """Exploratory ML factor analysis with varimax and deterministic sign orientation."""
    from sklearn.decomposition import FactorAnalysis
    from sklearn.exceptions import ConvergenceWarning

    clean = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    dropped_columns = clean.columns[clean.std(ddof=0).fillna(0) < 1e-10].tolist()
    clean = clean.drop(columns=dropped_columns)
    n, p = clean.shape
    if p < 3 or n_factors not in (2, 3) or n_factors >= p:
        raise ValueError("因子分析至少需要 3 个有变化的变量，且因子数必须小于变量数。")
    if n < max(30, p * 5):
        raise ValueError(f"完整样本 {n} 位；探索性因子分析至少需要 {max(30, p * 5)} 位。")
    z = (clean - clean.mean()) / clean.std(ddof=0)
    if np.linalg.matrix_rank(z) < p:
        raise ValueError("变量完全共线，请先删除重复或线性依赖的变量。")
    model = FactorAnalysis(n_components=n_factors, rotation="varimax", svd_method="lapack", max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            scores = model.fit_transform(z)
        except ConvergenceWarning as error:
            raise ValueError("因子模型未收敛，请调整变量或增加有效样本。") from error
    loadings = model.components_.T.copy()
    for i in range(n_factors):
        sign = np.sign(loadings[np.argmax(np.abs(loadings[:, i])), i]) or 1
        loadings[:, i] *= sign
        scores[:, i] *= sign
    labels = [f"因子 {i + 1}" for i in range(n_factors)]
    covariance = z.to_numpy().T @ z.to_numpy() / n
    residual = covariance - (loadings @ loadings.T + np.diag(model.noise_variance_))
    return dict(scores=pd.DataFrame(scores, index=clean.index, columns=labels),
                loadings=pd.DataFrame(loadings, index=clean.columns, columns=labels),
                communalities=pd.Series((loadings ** 2).sum(axis=1), index=clean.columns),
                uniqueness=pd.Series(model.noise_variance_, index=clean.columns),
                correlation=z.corr(), n=n, dropped_rows=len(frame) - n,
                dropped_columns=dropped_columns, condition=float(np.linalg.cond(covariance)),
                residual=float(np.linalg.norm(residual) / np.linalg.norm(covariance)))
