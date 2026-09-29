import io
import json

import numpy as np
import pandas as pd
import pytest

from data_source import build_dataframe_data_source
from intent_subsystem import IntentEngine, IntentModel, ALL_FEATURE_NAMES
from workbench_data import demo_source, parse_boolean, read_upload, source_tables, validate_table


@pytest.mark.parametrize("value,expected", [("false", False), ("False", False), (0, False),
        ("no", False), (None, False), ("true", True), (1, True), ("是", True)])
def test_boolean_parsing(value, expected):
    assert parse_boolean(value) is expected


def test_demo_is_reproducible_without_changing_global_random_state():
    np.random.seed(32)
    expected = np.random.random()
    np.random.seed(32)
    a, b = demo_source(6), demo_source(6)
    assert np.random.random() == expected
    pd.testing.assert_frame_equal(source_tables(a)[0], source_tables(b)[0])
    assert len(a.list_customers()) == 6


def test_import_validates_schema_and_false_and_cross_references():
    customers, products = source_tables(demo_source(3))
    customers["first_buy"] = "false"
    source = build_dataframe_data_source(customers, products)
    assert not source.get_customer("DEMO_001")["first_buy"]
    for invalid in [customers.drop(columns="risk"), customers.assign(risk="C9"),
                    customers.assign(age=4.5), customers.assign(assets=np.inf),
                    pd.concat([customers, customers.iloc[:1]])]:
        with pytest.raises(ValueError):
            build_dataframe_data_source(invalid, products)
    intent = pd.DataFrame([dict(customer_id="UNKNOWN", feature_name="beh_view_cnt_7d", feature_value=3)])
    with pytest.raises(ValueError, match="不存在"):
        build_dataframe_data_source(customers, products, intent)


def test_csv_json_and_excel_import_roundtrip():
    customers, products = source_tables(demo_source(3))
    tables = read_upload("customers.csv", customers.to_csv(index=False).encode())
    assert len(tables["customers"]) == 3
    content = json.dumps({"customers": customers.to_dict("records"), "products": products.to_dict("records")}).encode()
    assert set(read_upload("data.json", content)) == {"customers", "products"}
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        customers.to_excel(writer, sheet_name="customers", index=False)
        products.to_excel(writer, sheet_name="products", index=False)
    assert set(read_upload("data.xlsx", output.getvalue())) == {"customers", "products"}


def test_partial_intent_features_are_completed_before_prediction():
    class RecordingModel:
        is_trained = True
        feature_names = ALL_FEATURE_NAMES

        def predict_proba(self, frame):
            assert frame.shape == (1, len(ALL_FEATURE_NAMES))
            assert frame.iloc[0]["beh_view_cnt_7d"] == 3
            assert not frame.isna().any().any()
            return np.array([[.4, .6]])

        def explain(self, features, top_k):
            raise RuntimeError("optional explainer unavailable")

    result = IntentEngine(RecordingModel()).fused_intent_score_from_features({"beh_view_cnt_7d": 3})
    assert result["model_score"] == .6
    assert result["top_signals"] == []
    assert IntentModel().explain(dict.fromkeys(ALL_FEATURE_NAMES, 0)) == []
