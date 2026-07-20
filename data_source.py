# data_source.py
"""
数据源抽象层 - 支持模拟 / SQLite / MySQL / PostgreSQL / 内存数据（文件上传）
"""
import sqlite3
from typing import Dict, List, Optional
from abc import ABC, abstractmethod


# ================================================================
# 异常定义
# ================================================================

class DataSourceError(Exception):
    pass


class DataSourceConnectionError(DataSourceError):
    pass


class DataSourceNotFoundError(DataSourceError):
    pass


# ================================================================
# 1. 数据源基类
# ================================================================

class DataSource(ABC):
    @abstractmethod
    def get_customer(self, customer_id: str) -> Dict:
        pass

    @abstractmethod
    def get_product(self, product_id: str) -> Dict:
        pass

    @abstractmethod
    def list_customers(self) -> List[str]:
        pass

    @abstractmethod
    def list_products(self) -> List[str]:
        pass

    @abstractmethod
    def get_intent_features(self, customer_id: str, product_id: Optional[str] = None) -> Dict[str, float]:
        pass


# ================================================================
# 2. 模拟数据源
# ================================================================

class MockDataSource(DataSource):
    def __init__(self):
        self._customers = {
            "CUST_HIGH": {"risk": "C3", "age": 42, "assets": 2000000, "period": 365,
                          "first_buy": False, "name": "张先生", "income": "高"},
            "CUST_LOW": {"risk": "C2", "age": 28, "assets": 50000, "period": 90,
                         "first_buy": True, "name": "李女士", "income": "中"},
            "CUST_ELDER": {"risk": "C4", "age": 72, "assets": 1200000, "period": 730,
                           "first_buy": False, "name": "王先生", "income": "高"},
        }
        self._products = {
            "P001": {"risk": "R1", "name": "天天利货币", "lock": 0, "min": 0, "type": "货币型"},
            "P002": {"risk": "R2", "name": "季季盈固收", "lock": 90, "min": 10000, "type": "债券型"},
            "P004": {"risk": "R3", "name": "一年平衡混合", "lock": 365, "min": 100000, "type": "混合型"},
            "P005": {"risk": "R4", "name": "进取科创主题", "lock": 730, "min": 200000, "type": "股票型"},
            "P006": {"risk": "R5", "name": "全球精选权益", "lock": 1095, "min": 500000, "type": "股票型"},
        }
        self._intent_features = {
            "CUST_HIGH": {"beh_view_cnt_7d": 12.0, "beh_view_duration_decay": 110.0,
                          "beh_calculator_use_cnt": 4.0, "beh_compare_cnt": 7.0,
                          "beh_revisit_gap_avg": 2.5, "txn_history_product_types": 3.0,
                          "txn_redemption_freq": 0.3, "txn_avg_holding_period": 180.0,
                          "txn_days_since_last_purchase": 5.0, "profile_aum_tier": 4.0,
                          "profile_lifecycle_stage": 3.0, "profile_risk_level": 3.0,
                          "interact_push_open_rate_30d": 0.8, "interact_advisor_contact_freq": 1.0,
                          "interact_last_script_accepted": 1.0},
            "CUST_LOW": {"beh_view_cnt_7d": 0.0, "beh_view_duration_decay": 1.0,
                         "beh_calculator_use_cnt": 0.0, "beh_compare_cnt": 0.0,
                         "beh_revisit_gap_avg": 200.0, "txn_history_product_types": 0.0,
                         "txn_redemption_freq": 0.0, "txn_avg_holding_period": 0.0,
                         "txn_days_since_last_purchase": 180.0, "profile_aum_tier": 1.0,
                         "profile_lifecycle_stage": 1.0, "profile_risk_level": 2.0,
                         "interact_push_open_rate_30d": 0.0, "interact_advisor_contact_freq": 0.0,
                         "interact_last_script_accepted": 0.0},
            "CUST_ELDER": {"beh_view_cnt_7d": 4.0, "beh_view_duration_decay": 35.0,
                           "beh_calculator_use_cnt": 1.0, "beh_compare_cnt": 1.0,
                           "beh_revisit_gap_avg": 12.0, "txn_history_product_types": 4.0,
                           "txn_redemption_freq": 0.2, "txn_avg_holding_period": 300.0,
                           "txn_days_since_last_purchase": 20.0, "profile_aum_tier": 3.0,
                           "profile_lifecycle_stage": 2.0, "profile_risk_level": 4.0,
                           "interact_push_open_rate_30d": 0.3, "interact_advisor_contact_freq": 0.0,
                           "interact_last_script_accepted": 0.0},
        }

    def get_customer(self, customer_id: str) -> Dict:
        return self._customers.get(customer_id, {})

    def get_product(self, product_id: str) -> Dict:
        return self._products.get(product_id, {})

    def list_customers(self) -> List[str]:
        return list(self._customers.keys())

    def list_products(self) -> List[str]:
        return list(self._products.keys())

    def get_intent_features(self, customer_id: str, product_id: Optional[str] = None) -> Dict[str, float]:
        return self._intent_features.get(customer_id, {})


# ================================================================
# 3. SQLite 数据源
# ================================================================

class SQLiteDataSource(DataSource):
    def __init__(self, db_path: str = "prudence.db"):
        self.db_path = db_path
        self._ensure_tables()

    def _ensure_tables(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS customers
                     (id TEXT PRIMARY KEY, risk TEXT, age INTEGER, assets INTEGER, period INTEGER,
                      first_buy INTEGER, name TEXT, income TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS products
                     (id TEXT PRIMARY KEY, risk TEXT, name TEXT, lock INTEGER, min INTEGER, type TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS intent_features
                     (customer_id TEXT, feature_name TEXT, feature_value REAL,
                      PRIMARY KEY (customer_id, feature_name))''')
        c.execute("SELECT COUNT(*) FROM customers")
        if c.fetchone()[0] == 0:
            self._insert_sample_data(c)
        conn.commit()
        conn.close()

    def _insert_sample_data(self, c):
        sample_customers = [
            ('CUST_HIGH', 'C3', 42, 2000000, 365, 0, '张先生', '高'),
            ('CUST_LOW', 'C2', 28, 50000, 90, 1, '李女士', '中'),
            ('CUST_ELDER', 'C4', 72, 1200000, 730, 0, '王先生', '高'),
        ]
        c.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?,?,?)", sample_customers)

        sample_products = [
            ('P001', 'R1', '天天利货币', 0, 0, '货币型'),
            ('P002', 'R2', '季季盈固收', 90, 10000, '债券型'),
            ('P004', 'R3', '一年平衡混合', 365, 100000, '混合型'),
            ('P005', 'R4', '进取科创主题', 730, 200000, '股票型'),
            ('P006', 'R5', '全球精选权益', 1095, 500000, '股票型'),
        ]
        c.executemany("INSERT INTO products VALUES (?,?,?,?,?,?)", sample_products)

        mock = MockDataSource()
        for cid in mock.list_customers():
            for fname, fval in mock.get_intent_features(cid).items():
                c.execute("INSERT INTO intent_features VALUES (?,?,?)", (cid, fname, fval))

    def get_customer(self, customer_id: str) -> Dict:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM customers WHERE id=?", (customer_id,))
        row = c.fetchone()
        conn.close()
        if row:
            return {"risk": row[1], "age": row[2], "assets": row[3], "period": row[4],
                    "first_buy": bool(row[5]), "name": row[6], "income": row[7]}
        return {}

    def get_product(self, product_id: str) -> Dict:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM products WHERE id=?", (product_id,))
        row = c.fetchone()
        conn.close()
        if row:
            return {"risk": row[1], "name": row[2], "lock": row[3], "min": row[4], "type": row[5]}
        return {}

    def list_customers(self) -> List[str]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id FROM customers")
        rows = c.fetchall()
        conn.close()
        return [r[0] for r in rows]

    def list_products(self) -> List[str]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id FROM products")
        rows = c.fetchall()
        conn.close()
        return [r[0] for r in rows]

    def get_intent_features(self, customer_id: str, product_id: Optional[str] = None) -> Dict[str, float]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT feature_name, feature_value FROM intent_features WHERE customer_id=?", (customer_id,))
        rows = c.fetchall()
        conn.close()
        return {r[0]: r[1] for r in rows}


# ================================================================
# 4. MySQL / PostgreSQL 数据源（使用 SQLAlchemy）
# ================================================================

class SQLAlchemyDataSource(DataSource):
    def __init__(self, connection_string: str):
        from sqlalchemy import create_engine, text
        self.engine = create_engine(connection_string)
        self.text = text

    def _row_to_dict(self, row):
        if row:
            return {key: value for key, value in row._mapping.items()}
        return {}

    def get_customer(self, customer_id: str) -> Dict:
        with self.engine.connect() as conn:
            result = conn.execute(self.text("SELECT * FROM customers WHERE id=:id"), {"id": customer_id})
            row = result.fetchone()
            return self._row_to_dict(row)

    def get_product(self, product_id: str) -> Dict:
        with self.engine.connect() as conn:
            result = conn.execute(self.text("SELECT * FROM products WHERE id=:id"), {"id": product_id})
            row = result.fetchone()
            return self._row_to_dict(row)

    def list_customers(self) -> List[str]:
        with self.engine.connect() as conn:
            result = conn.execute(self.text("SELECT id FROM customers"))
            return [r[0] for r in result]

    def list_products(self) -> List[str]:
        with self.engine.connect() as conn:
            result = conn.execute(self.text("SELECT id FROM products"))
            return [r[0] for r in result]

    def get_intent_features(self, customer_id: str, product_id: Optional[str] = None) -> Dict[str, float]:
        with self.engine.connect() as conn:
            result = conn.execute(
                self.text("SELECT feature_name, feature_value FROM intent_features WHERE customer_id=:cid"),
                {"cid": customer_id}
            )
            return {r[0]: r[1] for r in result}


# ================================================================
# 5. 内存数据源（用于文件上传）
# ================================================================

class MemoryDataSource(DataSource):
    def __init__(self, customers: Dict = None, products: Dict = None, intent_features: Dict = None):
        self._customers = customers or {}
        self._products = products or {}
        self._intent_features = intent_features or {}

    def set_customers(self, customers: Dict):
        self._customers = customers

    def set_products(self, products: Dict):
        self._products = products

    def set_intent_features(self, intent_features: Dict):
        self._intent_features = intent_features

    def get_customer(self, customer_id: str) -> Dict:
        return self._customers.get(customer_id, {})

    def get_product(self, product_id: str) -> Dict:
        return self._products.get(product_id, {})

    def list_customers(self) -> List[str]:
        return list(self._customers.keys())

    def list_products(self) -> List[str]:
        return list(self._products.keys())

    def get_intent_features(self, customer_id: str, product_id: Optional[str] = None) -> Dict[str, float]:
        return self._intent_features.get(customer_id, {})


# ================================================================
# 6. 数据源工厂
# ================================================================

def create_data_source(source_type: str, **kwargs) -> DataSource:
    source_type = source_type.lower()
    if source_type in ["mock", "simulate", "模拟", "模拟数据"]:
        return MockDataSource()
    elif source_type in ["sqlite", "sqlite3"]:
        db_path = kwargs.get("db_path", "prudence.db")
        return SQLiteDataSource(db_path)
    elif source_type in ["mysql", "mariadb"]:
        host = kwargs.get("host", "localhost")
        port = kwargs.get("port", 3306)
        database = kwargs.get("database", "prudence")
        user = kwargs.get("user", "root")
        password = kwargs.get("password", "")
        conn_str = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        return SQLAlchemyDataSource(conn_str)
    elif source_type in ["postgresql", "postgres", "pg"]:
        host = kwargs.get("host", "localhost")
        port = kwargs.get("port", 5432)
        database = kwargs.get("database", "prudence")
        user = kwargs.get("user", "postgres")
        password = kwargs.get("password", "")
        conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        return SQLAlchemyDataSource(conn_str)
    elif source_type in ["memory", "内存"]:
        customers = kwargs.get("customers", {})
        products = kwargs.get("products", {})
        intent_features = kwargs.get("intent_features", {})
        return MemoryDataSource(customers, products, intent_features)
    else:
        raise ValueError(f"不支持的数据源类型: {source_type}")


# ================================================================
# 7. 核心函数：从DataFrame构建数据源（修复导入错误的关键）
# ================================================================

def build_dataframe_data_source(
    customers_df=None,
    products_df=None,
    intent_df=None
) -> MemoryDataSource:
    """
    从Pandas DataFrame构建内存数据源
    用于文件上传场景
    """
    import pandas as pd

    customers = {}
    if customers_df is not None and not customers_df.empty:
        for _, row in customers_df.iterrows():
            if all(k in row for k in ['id', 'risk', 'age', 'assets', 'period']):
                cid = str(row['id'])
                customers[cid] = {
                    'risk': str(row['risk']),
                    'age': int(row['age']),
                    'assets': float(row['assets']),
                    'period': int(row['period']),
                    'first_buy': bool(row.get('first_buy', False)),
                    'name': str(row.get('name', '')),
                    'income': str(row.get('income', ''))
                }

    products = {}
    if products_df is not None and not products_df.empty:
        for _, row in products_df.iterrows():
            if all(k in row for k in ['id', 'risk', 'name', 'lock', 'min']):
                pid = str(row['id'])
                products[pid] = {
                    'risk': str(row['risk']),
                    'name': str(row['name']),
                    'lock': int(row['lock']),
                    'min': float(row['min']),
                    'type': str(row.get('type', ''))
                }

    intent_features = {}
    if intent_df is not None and not intent_df.empty:
        for _, row in intent_df.iterrows():
            cid = str(row.get('customer_id', ''))
            if cid:
                fname = row.get('feature_name')
                fval = row.get('feature_value')
                if fname is not None and fval is not None:
                    intent_features.setdefault(cid, {})[fname] = float(fval)

    return MemoryDataSource(customers, products, intent_features)


# ================================================================
# 8. 测试入口
# ================================================================

if __name__ == "__main__":
    print("✅ data_source.py 加载成功")
    ds = create_data_source("mock")
    print(f"模拟数据源客户: {ds.list_customers()}")