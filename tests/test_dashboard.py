from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from config import AppConfig
from main import PrudenceAPI
from workbench_data import demo_source


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    config = AppConfig()
    config.intent.model_path = str(tmp_path_factory.mktemp("models") / "intent.pkl")
    config.intent.max_training_samples = 300
    return PrudenceAPI(config, data_source=demo_source(size=48))


def app(api):
    at = AppTest.from_file(str(Path(__file__).parents[1] / "ui.py"), default_timeout=45)
    at.session_state["api"] = api
    at.session_state["source_label"] = "模拟测试数据"
    return at.run()


def click(at, label):
    next(button for button in at.button if button.label == label).click().run()
    assert not at.exception, [e.message for e in at.exception]


def test_integrated_dashboard_views_profile_and_filters(api):
    at = app(api)
    assert not at.exception, [e.message for e in at.exception]
    assert at.metric[0].value == "48"
    assert len(at.tabs) == 6
    for mode in ["关注度 × 年度贡献", "因子 1 × 因子 2", "关注度 × 匹配度"]:
        at.radio(key="scatter_mode").set_value(mode).run()
        assert not at.exception, [e.message for e in at.exception]
    at.selectbox(key="focus_customer").set_value("DEMO_002").run()
    assert at.session_state["focus_customer"] == "DEMO_002"
    at.text_input(key="customer_search").set_value("DEMO_001").run()
    assert at.metric[0].value == "1"
    assert at.session_state["focus_customer"] == "DEMO_001"
    at.text_input(key="customer_search").set_value("DOES_NOT_EXIST").run()
    assert at.metric[0].value == "0" and not at.exception


def test_point_selection_uses_customer_id_and_manual_fallback(api):
    at = app(api)
    key = at.session_state["scatter_key"]
    at.session_state[key] = {"selection": {"points": [{"point_index": 0, "curve_number": 2, "customdata": ["DEMO_004"]}]}}
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert at.session_state["focus_customer"] == "DEMO_004"
    assert at.session_state["selected_customers"] == ["DEMO_004"]
    at.selectbox(key="focus_customer").set_value("DEMO_006").run()
    assert at.session_state["focus_customer"] == "DEMO_006"
    at.selectbox(key="product_id").set_value("P002").run()
    assert at.session_state["selected_customers"] == []


def test_scenario_edits_and_plan_invalidation(api):
    at = app(api)
    at.selectbox(key="product_id").set_value("P002").run()
    old_contribution = at.metric[3].value
    at.number_input(key="rate_deposit_rate_P002").set_value(3.)
    click(at, "应用利率与策略")
    assert at.metric[3].value != old_contribution
    click(at, "生成服务计划")
    selected = at.session_state["allocation"][1].selected
    assert selected.suitability_level.eq("ALLOW").all()
    assert not selected.customer_id.duplicated().any()
    at.number_input(key="allocation_budget").set_value(0.).run()
    assert any("参数已改变" in item.value for item in at.info)
    click(at, "生成服务计划")
    assert at.session_state["allocation"][1].selected.empty
    at.slider(key="ds_P002").set_value(90)
    click(at, "应用利率与策略")
    assert any("100%" in item.value for item in at.error)
    assert at.session_state["scenario_settings"]["rates"]["deposit_share"] == .6
    at.slider(key="ds_P002").set_value(60)
    at.slider(key="ws_P002").set_value(0)
    click(at, "应用利率与策略")
    at.number_input(key="allocation_budget").set_value(1000.).run()
    click(at, "生成服务计划")
    assert at.session_state["allocation"][1].selected.empty


def test_csv_export_preserves_chinese_and_neutralizes_formulas():
    from dashboard import csv_bytes
    output = csv_bytes(pd.DataFrame({"客户": ["=1+2", " +cmd", "张三"]})).decode("utf-8-sig")
    assert "'=1+2" in output and "' +cmd" in output and "张三" in output


def test_ai_is_explicit_and_results_follow_current_context(api, monkeypatch):
    import dashboard_ai
    calls = []
    def fake_explain(settings, context, question):
        calls.append((settings, context, question))
        return dict(text="这是模拟接口返回的测试解读。", model=settings.model, protocol=settings.protocol,
                    endpoint="api.openai.com", elapsed_seconds=.01, usage={"total_tokens": 100}, truncated=False)
    monkeypatch.setattr(dashboard_ai, "explain", fake_explain)
    at = app(api)
    assert not calls and at.button(key="ai_generate").disabled
    at.text_input(key="ai_model_0").set_value("test-model").run()
    at.text_input(key="ai_key_0").set_value("fake-ui-key").run()
    assert not calls
    click(at, "发送摘要并生成解读")
    assert len(calls) == 1 and "selected_customer" not in calls[0][1]
    assert any("模拟接口返回" in t.value for t in at.text)
    at.radio(key="ai_scope").set_value("所选客户画像").run()
    assert any("旧解读" in i.value for i in at.info)
    click(at, "发送摘要并生成解读")
    assert len(calls) == 2 and "selected_customer" in calls[1][1]
    at.text_input(key="ai_base_0").set_value("https://other.example/v1").run()
    assert at.text_input(key="ai_key_0").value == ""
    assert at.button(key="ai_generate").disabled
    assert not at.exception
