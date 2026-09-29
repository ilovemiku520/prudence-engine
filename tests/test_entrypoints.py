"""Launch paths must not train models or substitute demonstration data."""
from types import SimpleNamespace
import sys

from fastapi.testclient import TestClient
import pytest

import api
import main
from config import AppConfig
from data_source import DataSourceError, MemoryDataSource


@pytest.mark.parametrize("mode", ["api", "ui"])
def test_launch_does_not_construct_decision_engine(monkeypatch, mode):
    def unexpected(*args, **kwargs):
        raise AssertionError("Launching a server must not eagerly train a model")
    monkeypatch.setattr(main, "PrudenceAPI", unexpected)
    monkeypatch.setattr(main, "StructuredLogger", lambda **kwargs: SimpleNamespace(info=lambda *a: None))
    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", mode])
    seen = []
    if mode == "api":
        import uvicorn
        monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: seen.append((a, kw)))
    else:
        import subprocess
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: seen.append((a, kw)))
    main.main()
    assert len(seen) == 1
    if mode == "ui":
        assert seen[0][0][0][:3] == [sys.executable, "-m", "streamlit"]
        assert seen[0][1]["check"] is True


def test_missing_explicit_config_is_not_silently_ignored(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "api", "--config", str(tmp_path / "missing.json")])
    with pytest.raises(SystemExit) as error:
        main.main()
    assert error.value.code == 2
    assert "配置文件不存在" in capsys.readouterr().err


def test_product_loading_failure_does_not_substitute_demo_catalog():
    class BrokenSource(MemoryDataSource):
        def list_products(self):
            raise DataSourceError("catalog unavailable")
    with pytest.raises(DataSourceError, match="catalog unavailable"):
        main.PrudenceFactory.create_engines(AppConfig(), BrokenSource())


def test_empty_api_source_stays_empty_and_unknown_details_are_404(monkeypatch):
    monkeypatch.setattr(api, "PrudenceAPI", lambda config: SimpleNamespace(data_source=MemoryDataSource()))
    config = AppConfig()
    config.api.admin_token = "test-admin"
    client = TestClient(api.create_app(config))
    headers = {"X-Admin-Token": "test-admin"}
    assert client.get("/api/customers", headers=headers).json() == {"customers": [], "total": 0}
    assert client.get("/api/products").json() == {"products": [], "total": 0}
    assert client.get("/api/customer/UNKNOWN", headers=headers).status_code == 404
    assert client.get("/api/product/UNKNOWN").status_code == 404
