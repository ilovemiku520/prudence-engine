import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api import BatchDecisionRequest, DecisionRequest, create_app
from config import AppConfig
from data_source import MemoryDataSource
from logger import StructuredLogger, clear_audit_logs, get_audit_logs
from nexus_orchestrator import NexusFeatureStore, NexusOrchestrator


class _NeverCalledDecision:
    def decide(self, customer, product):
        raise AssertionError("unknown identifiers must fail before decision evaluation")


def test_unknown_customer_is_not_replaced_with_default_profile():
    orchestrator = NexusOrchestrator(
        data_source=MemoryDataSource(customers={}, products={}),
        feature_store=NexusFeatureStore(enabled=False),
        suitability_engine=object(),
        intent_engine=object(),
        decision_engine=_NeverCalledDecision(),
    )

    result = orchestrator.orchestrate("UNKNOWN", "UNKNOWN")

    assert result["action"] == "ERROR"
    assert "拒绝使用默认画像" in result["reason"]


def test_audit_log_pseudonymizes_identifiers(tmp_path):
    clear_audit_logs()
    logger = StructuredLogger(name="test-audit", log_dir=str(tmp_path))
    logger.audit("decision", customer_id="CUST_SECRET", product_id="P_SECRET")

    record = get_audit_logs(1)[0]
    assert record["customer_id"].startswith("sha256:")
    assert record["product_id"].startswith("sha256:")
    assert "CUST_SECRET" not in str(record)


def test_batch_request_has_hard_limit():
    requests = [DecisionRequest(customer_id=f"CUST_{i}", product_id="P001") for i in range(101)]

    with pytest.raises(ValidationError):
        BatchDecisionRequest(requests=requests)


def test_admin_endpoints_are_closed_without_token():
    config = AppConfig()
    config.api.admin_token = ""
    client = TestClient(create_app(config))

    response = client.get("/api/audit")

    assert response.status_code == 503
