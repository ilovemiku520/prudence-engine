"""Session-local data and suitability rules without the optional prediction stack."""
from dataclasses import dataclass

from config import AppConfig
from data_source import DataSource, create_data_source
from prudence_suitability import SuitabilityEngine, SuitabilityMatrixConfig


@dataclass
class DashboardContext:
    data_source: DataSource
    suitability: SuitabilityEngine

    @classmethod
    def create(cls, config: AppConfig, source=None):
        if source is None:
            ds = config.data_source
            source = create_data_source(ds.type, host=ds.host, port=ds.port,
                database=ds.database, user=ds.user, password=ds.password, db_path=ds.db_path)
        catalog = []
        for pid in source.list_products():
            product = source.get_product(pid)
            if product:
                catalog.append(dict(id=pid, risk_level=product["risk"], name=product.get("name", ""),
                    lock_period=product["lock"], min_amount=product["min"]))
        return cls(source, SuitabilityEngine(SuitabilityMatrixConfig(), catalog))
