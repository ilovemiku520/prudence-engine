"""Import validation and reproducible, explicitly synthetic workbench samples."""
import io
import json
import re
from zipfile import BadZipFile

import numpy as np
import pandas as pd


def parse_boolean(value):
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0", "yes", "是"}:
        return True
    if text in {"false", "0", "0.0", "no", "否", ""}:
        return False
    raise ValueError("first_buy 必须为 true/false、1/0 或 是/否。")


def validate_table(frame, kind):
    required = {
        "customers": ["id", "risk", "age", "assets", "period"],
        "products": ["id", "risk", "name", "lock", "min"],
        "intent_features": ["customer_id", "feature_name", "feature_value"],
    }[kind]
    if frame is None:
        return None
    frame = frame.copy()
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{kind} 缺少字段：{', '.join(sorted(missing))}")
    if len(frame) > 20_000:
        raise ValueError("每张导入表最多 20,000 行。")
    if frame[required].isna().any().any():
        raise ValueError(f"{kind} 的必填字段存在空值。")
    id_column = "customer_id" if kind == "intent_features" else "id"
    frame[id_column] = frame[id_column].astype(str).str.strip()
    if not frame[id_column].map(lambda x: bool(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", x))).all():
        raise ValueError("标识必须为 1–64 位字母、数字、下划线或连字符。")
    keys = ["customer_id", "feature_name"] if kind == "intent_features" else ["id"]
    if frame.duplicated(keys).any():
        raise ValueError(f"{kind} 存在重复标识或重复特征。")
    if kind != "intent_features":
        prefix = "C" if kind == "customers" else "R"
        frame["risk"] = frame.risk.astype(str).str.strip().str.upper()
        if not frame.risk.isin([f"{prefix}{i}" for i in range(1, 6)]).all():
            raise ValueError(f"风险等级必须为 {prefix}1–{prefix}5。")
    numeric = {"customers": ["age", "assets", "period"],
               "products": ["lock", "min"], "intent_features": ["feature_value"]}[kind]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column]).all() or (frame[column] < 0).any():
            raise ValueError(f"{column} 必须是非负有限数值。")
        if column in {"age", "period", "lock"} and (frame[column] % 1 != 0).any():
            raise ValueError(f"{column} 必须是整数。")
    if kind == "customers":
        if (frame.age > 120).any():
            raise ValueError("age 必须在 0–120 之间。")
        frame["first_buy"] = frame.get("first_buy", pd.Series(False, index=frame.index)).map(parse_boolean)
    if kind == "intent_features":
        from intent_subsystem import ALL_FEATURE_NAMES
        if not frame.feature_name.isin(ALL_FEATURE_NAMES).all():
            raise ValueError("意图表包含未注册的 feature_name，请使用示例中的特征名。")
    return frame


def read_upload(name, content):
    if len(content) > 10 * 1024 * 1024:
        raise ValueError("文件不能超过 10 MB。")
    suffix = name.rsplit(".", 1)[-1].lower()
    if suffix == "json":
        payload = json.loads(content.decode("utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("JSON 顶层必须为包含 customers/products/intent_features 的对象。")
        tables = {key: pd.DataFrame(payload[key]) for key in
                  ("customers", "products", "intent_features") if key in payload}
    else:
        try:
            frames = [pd.read_csv(io.BytesIO(content))] if suffix == "csv" else (
                list(pd.read_excel(io.BytesIO(content), sheet_name=None).values()) if suffix == "xlsx" else [])
        except BadZipFile as error:
            raise ValueError("Excel 文件损坏或不是有效的 .xlsx 文件。") from error
        tables = {}
        for frame in frames:
            kind = ("intent_features" if "feature_name" in frame else
                    "products" if "lock" in frame else "customers")
            if kind in tables:
                raise ValueError(f"同一文件中有多张 {kind} 表，请合并后导入。")
            tables[kind] = frame
    if not tables:
        raise ValueError("未找到可导入的数据表。")
    return {key: validate_table(frame, key) for key, frame in tables.items()}


def source_tables(source):
    customers = pd.DataFrame([dict(source.get_customer(cid), id=cid) for cid in source.list_customers()])
    products = pd.DataFrame([dict(source.get_product(pid), id=pid) for pid in source.list_products()])
    return customers, products


def demo_source(size=120, seed=42):
    from data_source import MemoryDataSource, MockDataSource
    from intent_subsystem import ALL_FEATURE_NAMES
    rng = np.random.default_rng(seed)
    base = MockDataSource()
    customers, features = {}, {}
    for i in range(size):
        cid = f"DEMO_{i + 1:03d}"
        risk = int(rng.integers(1, 6))
        age = int(rng.integers(22, 80))
        assets = float(np.round(rng.lognormal(12.5 + risk * .12, .8), -3))
        customers[cid] = dict(risk=f"C{risk}", age=age, assets=assets,
                              period=int(rng.choice([90, 180, 365, 730, 1095])),
                              first_buy=bool(rng.random() < .25), name=f"模拟客户 {i + 1:03d}", income="模拟")
        row = {name: 0.0 for name in ALL_FEATURE_NAMES}
        engagement = rng.uniform(.3, 3)
        row.update(beh_view_cnt_7d=float(rng.poisson(engagement * 4)),
                   beh_view_duration_decay=float(rng.exponential(engagement * 25)),
                   beh_calculator_use_cnt=float(rng.poisson(engagement)),
                   beh_compare_cnt=float(rng.poisson(engagement * 1.5)),
                   beh_revisit_gap_avg=float(rng.uniform(1, 80)),
                   txn_days_since_last_purchase=float(rng.integers(1, 120)),
                   txn_avg_holding_period=float(rng.integers(30, 500)),
                   profile_aum_tier=float(np.clip(assets // 300000 + 1, 1, 5)),
                   profile_lifecycle_stage=3.0, profile_risk_level=float(risk),
                   interact_push_open_rate_30d=float(rng.beta(2, 4)))
        features[cid] = row
    return MemoryDataSource(customers, {pid: base.get_product(pid) for pid in base.list_products()}, features)
