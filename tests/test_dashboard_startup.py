"""The public dashboard must work without the prediction engine dependencies."""
import os
from pathlib import Path
import subprocess
import sys

from config import AppConfig
from dashboard_context import DashboardContext
from data_source import DataSourceError, MemoryDataSource
import pytest


def test_dashboard_and_import_do_not_load_or_train_prediction_stack(tmp_path):
    repo = Path(__file__).parents[1]
    script = r'''
import importlib.abc
import json
import os
from pathlib import Path
import sys

class NoPrediction(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'xgboost', 'shap', 'numba', 'main', 'intent_subsystem'}:
            raise AssertionError('Dashboard imported prediction stack: ' + fullname)
sys.meta_path.insert(0, NoPrediction())
from streamlit.testing.v1 import AppTest
from workbench_data import demo_source, source_tables, read_upload

at = AppTest.from_file('ui.py', default_timeout=45).run()
assert not at.exception, [e.message for e in at.exception]
assert at.metric[0].value == '120'
assert 'sklearn' not in sys.modules, 'Initial view must not fit a hidden factor model'
at.session_state['analysis_tab'] = '共同因子与稳定性'
at.run()
assert not at.exception, [e.message for e in at.exception]
assert any(m.label == '因子分析有效样本' for m in at.metric)
at.radio(key='scatter_mode').set_value('因子 1 × 因子 2').run()
assert not at.exception
at.session_state['profile_details'] = True
at.run()
assert not at.exception
source = demo_source(size=3)
c, p = source_tables(source)
payload = dict(customers=c.to_dict('records'), products=p.to_dict('records'),
    intent_features=[dict(customer_id=cid, feature_name=k, feature_value=v)
      for cid in source.list_customers() for k, v in source.get_intent_features(cid).items()])
read_upload('sample.json', json.dumps(payload).encode())
assert not Path(os.environ['INTENT_MODEL_PATH']).exists()
'''
    env = dict(os.environ, DS_TYPE="mock", INTENT_MODEL_PATH=str(tmp_path / "must-not-train.pkl"))
    result = subprocess.run([sys.executable, "-c", script], cwd=repo, env=env,
                            text=True, capture_output=True, timeout=75)
    assert result.returncode == 0, result.stdout + result.stderr


def test_dashboard_source_errors_and_empty_catalog_are_not_hidden():
    context = DashboardContext.create(AppConfig(), MemoryDataSource())
    assert context.data_source.list_products() == []

    class BrokenSource(MemoryDataSource):
        def list_products(self):
            raise DataSourceError("catalog unavailable")

    with pytest.raises(DataSourceError, match="catalog unavailable"):
        DashboardContext.create(AppConfig(), BrokenSource())
