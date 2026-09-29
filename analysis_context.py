"""Allowlisted, bounded numeric evidence for AI explanations (no names or IDs)."""
from dataclasses import asdict
import hashlib
import json
import math

from customer_scoring import DIMENSIONS, SIGNALS


def number(value):
    try:
        value = float(value)
        return round(value, 6) if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def build_context(frame, product, scenario, alpha, *, scope="overview", selected_id=None,
                  factors=None, factor_error="", synthetic=False):
    if scope not in ("overview", "customer", "factors"):
        raise ValueError("未知的 AI 解读范围。")
    if frame.empty:
        raise ValueError("当前没有客户数据可供解读。")
    result = {
        "scope": scope, "data_kind": "synthetic_demo" if synthetic else "user_data",
        "units": {"money": "CNY", "rates": "decimal_per_year", "scores": "0_to_100", "period": "days"},
        "product": {k: product[k] for k in ("risk", "lock", "min")},
        "scenario_assumptions": asdict(scenario), "attention_weight": alpha,
        "methods": {
            "score": "100*(engagement/100)^alpha*(fit/100)^(1-alpha); missing stays null",
            "dimensions": {
                "saturation": "S(x,a)=100*min(log(1+x)/log(1+a),1); anchors are unvalidated engineering conventions",
                "近期活跃": "[S(beh_view_cnt_7d,20)+100*2^(-txn_days_since_last_purchase/30)]/2",
                "决策投入": "[S(beh_calculator_use_cnt,5)+S(beh_compare_cnt,10)]/2",
                "互动响应": "50*interact_push_open_rate_30d+0.25*S(interact_advisor_contact_freq,5)+25*interact_last_script_accepted",
                "风险适配": "100*min(customer_risk_ordinal/product_risk_ordinal,1); display only, not the hard rule",
                "资金余量": "100*max(1-product_min/assets,0); zero assets -> zero",
                "期限匹配": "100*min(customer_period/product_lock,1); zero product_lock -> 100",
                "engagement": "mean(first 3 dimensions)", "fit": "mean(last 3 dimensions)",
            },
            "annual_contribution": "deposit*(FTP-deposit_rate)+loan*(loan_rate-FTP)+wealth*fee-loan*PD*LGD-annual_cost",
            "meaning": "observations and one-year constant-balance scenarios; not purchase probability, credit rating, actual profit or CLV",
            "suitability": "ALLOW / RESTRICTED / FORBID are deterministic hard rules; AI cannot override them",
        },
        "cohort": {"customers": len(frame), "complete_scores": int(frame.score.notna().sum()),
                   "suitability_counts": {level: int(frame.suitability_level.eq(level).sum()) for level in ("ALLOW", "RESTRICTED", "FORBID")},
                   "assumed_loan_balances": int(frame.loan_is_assumed.sum())},
        "distribution": {},
        "financial_totals": {},
    }
    for field in DIMENSIONS + ["engagement", "fit", "score", "annual_contribution"]:
        values = frame[field]
        result["distribution"][field] = dict(median=number(values.median()),
            q25=number(values.quantile(.25)), q75=number(values.quantile(.75)), missing=int(values.isna().sum()))
    for field in ("deposit_balance", "wealth_balance", "loan_balance", "unallocated", "deposit_contribution",
                  "loan_contribution", "wealth_contribution", "expected_loss", "annual_contribution", "customer_annual_cashflow"):
        result["financial_totals"][field] = number(frame[field].sum())
    if scope == "customer":
        if selected_id not in frame.index:
            raise ValueError("请先选择当前筛选范围内的客户。")
        row = frame.loc[selected_id]
        result["selected_customer"] = {field: number(row[field]) for field in DIMENSIONS + [
            "engagement", "fit", "score", "coverage", "assets", "period", "deposit_balance", "wealth_balance",
            "loan_balance", "unallocated", "deposit_contribution", "loan_contribution", "wealth_contribution",
            "expected_loss", "annual_contribution", "customer_annual_cashflow"]}
        result["selected_customer"].update(suitability_level=row.suitability_level,
            loan_is_assumed=bool(row.loan_is_assumed), risk_level=row["risk"])
        result["selected_customer"]["observed_signals"] = {key: number(row.signals.get(key)) for key in SIGNALS}
    if scope == "factors":
        if factors is None:
            result["factor_analysis"] = dict(available=False, note="样本或变量未满足因子模型计算条件，不能推断因子结果。")
        else:
            result["factor_analysis"] = dict(available=True, method="standardized ML factor analysis + Varimax; full-source fit, not filtered subset",
                sample_count=factors["n"], dropped_rows=factors["dropped_rows"],
                condition=number(factors["condition"]), relative_covariance_error=number(factors["residual"]),
                loadings={str(col): {str(k): number(v) for k, v in factors["loadings"][col].items()} for col in factors["loadings"]},
                communalities={str(k): number(v) for k, v in factors["communalities"].items()},
                uniqueness={str(k): number(v) for k, v in factors["uniqueness"].items()})
    return result


def context_signature(context, question, settings, local_scope=""):
    # Local scope may contain identities for invalidation; it is never sent to the provider.
    request = dict(context=context, question=question, protocol=settings.protocol, base_url=settings.base_url,
                   model=settings.model, max_tokens=settings.max_tokens, auth_mode=settings.auth_mode,
                   token_parameter=settings.token_parameter, local_scope=local_scope)
    return hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()
