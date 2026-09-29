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
    return PrudenceAPI(config, data_source=demo_source(size=12))


def app(api):
    at = AppTest.from_file(str(Path(__file__).parents[1] / "ui.py"), default_timeout=45)
    at.session_state["api"] = api
    at.session_state["source_label"] = "模拟测试数据"
    return at.run()


def navigate(at, page):
    at.radio(key="navigation").set_value(page).run()
    assert not at.exception, [e.message for e in at.exception]


def click(at, label):
    next(button for button in at.button if button.label == label).click().run()
    assert not at.exception, [e.message for e in at.exception]


def test_all_pages_and_editable_experiments(api):
    at = app(api)
    assert not at.exception
    assert at.metric[0].value == "12"
    for page in ["多元统计", "矩阵实验室", "运筹优化", "博弈实验", "数据管理"]:
        navigate(at, page)
    navigate(at, "矩阵实验室")
    next(r for r in at.radio if r.label == "矩阵来源").set_value("自定义矩阵").run()
    at.text_area[0].set_value("1, 2\n2, 4").run()
    assert not at.exception
    assert at.metric[0].value == "1"
    at.text_area[0].set_value("1 2\n3").run()
    assert at.error and not at.exception
    navigate(at, "博弈实验")
    next(r for r in at.radio if r.label == "实验类型").set_value("双矩阵博弈 · 纯策略 Nash").run()
    assert not at.exception
    assert "背离" in at.success[0].value


def test_decisions_feed_optimizer_and_parameter_changes_invalidate_result(api):
    at = app(api)
    navigate(at, "决策分析")
    click(at, "运行决策分析")
    assert len(at.session_state["batch"]) == 30
    assert all(item["action"] != "ERROR" for item in at.session_state["batch"])
    navigate(at, "运筹优化")
    click(at, "求解最优分配")
    selected = at.session_state["allocation"][1].selected
    assert selected.suitability_level.eq("ALLOW").all()
    assert not selected.customer_id.duplicated().any()
    at.number_input[0].set_value(0.0).run()
    assert any("参数已改变" in message.value for message in at.info)
    click(at, "求解最优分配")
    assert at.session_state["allocation"][1].selected.empty
    navigate(at, "决策分析")
    at.multiselect[0].set_value([]).run()
    click(at, "运行决策分析")
    assert at.error


def test_csv_export_preserves_chinese_and_neutralizes_formulas():
    from dashboard import csv_bytes
    output = csv_bytes(pd.DataFrame({"客户": ["=1+2", " +cmd", "张三"]})).decode("utf-8-sig")
    assert "'=1+2" in output and "' +cmd" in output and "张三" in output
