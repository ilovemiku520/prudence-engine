from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from customer_scoring import (DIMENSIONS, ValueScenario, composite, customer_scores, factor_inputs,
                              fit_factors, observed, scenario_values, selected_customer_ids)
from data_source import MemoryDataSource, build_dataframe_data_source
from prudence_suitability import SuitabilityEngine, SuitabilityMatrixConfig
from workbench_data import demo_source, source_tables, validate_table


def scores(source, pid="P004", alpha=.5):
    engine = SuitabilityEngine(SuitabilityMatrixConfig(), [])
    return customer_scores(source, engine, pid, source.get_product(pid), alpha)


def test_score_bounds_reproducibility_and_independence_from_cohort():
    source = demo_source()
    full = scores(source)
    assert full[DIMENSIONS + ["engagement", "fit", "score"]].ge(0).all().all()
    assert full[DIMENSIONS + ["engagement", "fit", "score"]].le(100).all().all()
    cid = full.index[5]
    subset = MemoryDataSource({cid: source.get_customer(cid)}, {"P004": source.get_product("P004")},
                              {cid: source.get_intent_features(cid)})
    pd.testing.assert_series_equal(scores(subset).loc[cid], full.loc[cid])
    pd.testing.assert_frame_equal(scores(demo_source()), full)


def test_absence_is_not_zero_and_invalid_signals_are_missing():
    source = demo_source(size=1)
    features = source.get_intent_features("DEMO_001")
    features.pop("beh_view_cnt_7d")
    features["interact_push_open_rate_30d"] = 1.1
    row = scores(source).iloc[0]
    assert np.isnan(row["近期活跃"]) and np.isnan(row["互动响应"]) and np.isnan(row.score)
    assert row.coverage == 4 and "浏览" in row.missing
    for value in [None, "bad", np.inf, -1]:
        assert np.isnan(observed(value))
    assert observed(0) == 0


def test_monotonic_attention_and_hard_gate_does_not_change_with_weights():
    source = demo_source(size=1)
    cid = "DEMO_001"
    source.get_customer(cid).update(risk="C1", age=35, period=1095, assets=1e6)
    before = scores(source).iloc[0]
    source.get_intent_features(cid)["beh_view_cnt_7d"] = 100
    after = scores(source).iloc[0]
    assert after.engagement >= before.engagement
    assert after.suitability_level == "FORBID"
    assert scores(source, alpha=.9).iloc[0].suitability_level == "FORBID"
    assert composite(0, 100) == 0


def test_scenario_reconciles_funds_profit_and_client_cashflow():
    frame = pd.DataFrame(dict(assets=[100000.], suitability_level=["ALLOW"], loan_balance=[50000.]))
    s = ValueScenario()
    row = scenario_values(frame, {"min": 10000.}, s).iloc[0]
    assert row.deposit_balance + row.wealth_balance + row.unallocated == pytest.approx(100000.)
    # 600 deposit spread + 750 loan spread + 200 fees - 225 losses - 100 cost.
    assert row.annual_contribution == pytest.approx(1225.)
    assert row.customer_annual_cashflow == pytest.approx(100.)
    assert not row.loan_is_assumed


@pytest.mark.parametrize("share,expected", [(.4, "ALLOW"), (.6, "RESTRICTED"), (.8, "FORBID")])
def test_planned_wealth_amount_tightens_existing_financial_rules(share, expected):
    source = demo_source(1)
    source.get_customer("DEMO_001").update(risk="C5", age=35, assets=1000000., period=1095, first_buy=False)
    engine = SuitabilityEngine(SuitabilityMatrixConfig(), [])
    frame = customer_scores(source, engine, "P002", source.get_product("P002"), planned_wealth_share=share)
    assert frame.iloc[0].suitability_level == expected


def test_negative_wealth_return_can_be_used_for_a_stress_scenario():
    scenario = replace(ValueScenario(), wealth_return=-.2)
    frame = pd.DataFrame(dict(assets=[100000.], suitability_level=["ALLOW"]))
    assert scenario_values(frame, {"min": 0}, scenario).iloc[0].customer_annual_cashflow < 0


@pytest.mark.parametrize("level,minimum", [("FORBID", 0), ("RESTRICTED", 0), ("ALLOW", 50000)])
def test_ineligible_or_below_minimum_wealth_is_never_allocated(level, minimum):
    row = scenario_values(pd.DataFrame(dict(assets=[100000.], suitability_level=[level])),
                          {"min": minimum}, ValueScenario()).iloc[0]
    assert row.wealth_balance == 0 and row.wealth_contribution == 0
    assert row.unallocated == 40000.
    assert row.loan_is_assumed and row.loan_balance == 0


def test_rate_sensitivity_and_separation_of_customer_return_from_bank_fee():
    frame = pd.DataFrame(dict(assets=[100000.], suitability_level=["ALLOW"], loan_balance=[50000.]))
    product, s = {"min": 0}, ValueScenario()
    base = scenario_values(frame, product, s).iloc[0]
    high_deposit = scenario_values(frame, product, replace(s, deposit_rate=s.deposit_rate + .01)).iloc[0]
    high_loan = scenario_values(frame, product, replace(s, loan_rate=s.loan_rate + .01)).iloc[0]
    high_return = scenario_values(frame, product, replace(s, wealth_return=.08)).iloc[0]
    assert high_deposit.annual_contribution == pytest.approx(base.annual_contribution - 600)
    assert high_loan.annual_contribution == pytest.approx(base.annual_contribution + 500)
    assert high_return.annual_contribution == base.annual_contribution
    assert high_return.customer_annual_cashflow > base.customer_annual_cashflow


@pytest.mark.parametrize("change", [{"deposit_share": .9}, {"loan_rate": -1}, {"annual_cost": np.inf}, {"default_probability": 1.1}])
def test_scenario_invalid_parameters_rejected(change):
    with pytest.raises(ValueError):
        replace(ValueScenario(), **change).validate()


def test_selection_uses_stable_ids_across_traces_and_rejects_stale_ids():
    event = {"selection": {"points": [
        {"point_index": 0, "curve_number": 2, "customdata": ["b", "B"]},
        {"point_index": 0, "curve_number": 0, "customdata": ["a", "A"]},
        {"customdata": ["b"]}, {"customdata": ["stale"]}, {"point_index": 1}]}}
    assert selected_customer_ids(event, ["a", "b"]) == ["b", "a"]
    assert selected_customer_ids(None, ["a"]) == []


def test_factor_analysis_recovers_shared_structure_and_reconstructs_covariance():
    rng = np.random.default_rng(7)
    latent = rng.normal(size=(400, 2))
    weights = np.array([[.9, 0], [.8, .1], [.85, 0], [0, .9], [.1, .85], [0, .8]])
    frame = pd.DataFrame(latent @ weights.T + rng.normal(0, .25, (400, 6)), columns=list("abcdef"))
    result = fit_factors(frame)
    loadings = result["loadings"].abs()
    assert loadings.loc[list("abc")].idxmax(axis=1).nunique() == 1
    assert loadings.loc[list("def")].idxmax(axis=1).nunique() == 1
    assert loadings.loc["a"].idxmax() != loadings.loc["d"].idxmax()
    assert result["residual"] < .05
    assert result["scores"].shape == (400, 2)
    assert np.allclose(result["communalities"] + result["uniqueness"], 1, atol=.03)
    pd.testing.assert_frame_equal(result["scores"], fit_factors(frame)["scores"])


def test_factor_analysis_reports_missing_rows_and_constant_variables():
    frame = factor_inputs(demo_source())
    frame.iloc[0, 0] = np.nan
    frame["constant"] = 5.
    result = fit_factors(frame)
    assert result["dropped_rows"] == 1 and result["n"] == 119
    assert result["dropped_columns"] == ["constant"]
    assert frame.index[0] not in result["scores"].index
    with pytest.raises(ValueError, match="完整样本"):
        fit_factors(frame.head(12))
    frame["duplicate"] = frame.iloc[:, 1]
    with pytest.raises(ValueError, match="共线"):
        fit_factors(frame)


def test_optional_loan_balance_import_roundtrip_and_validation():
    customers, products = source_tables(demo_source(3))
    customers["loan_balance"] = [150000., np.nan, 0.]
    source = build_dataframe_data_source(customers, products)
    assert source.get_customer("DEMO_001")["loan_balance"] == 150000.
    assert "loan_balance" not in source.get_customer("DEMO_002")
    customers.loc[0, "loan_balance"] = -100.
    with pytest.raises(ValueError, match="loan_balance"):
        validate_table(customers, "customers")
