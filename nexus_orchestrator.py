# nexus_orchestrator.py
"""
Nexus 编排层 - 协调数据源、特征存储、适当性、意图、融合决策
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import time
from collections import defaultdict
from loguru import logger

from intent_subsystem import ALL_FEATURE_NAMES
from data_source import DataSource


# ================================================================
# 特征存储
# ================================================================

class NexusFeatureStore:
    """特征存储 - 管理离线特征和实时特征"""

    def __init__(self, redis_host="localhost", redis_port=6379):
        self._offline_cache = {}
        self._realtime_cache = defaultdict(dict)
        self._redis = None

        try:
            import redis
            self._redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self._redis.ping()
            logger.info(f"Redis 连接成功: {redis_host}:{redis_port}")
        except Exception:
            logger.warning("Redis 不可用，使用内存缓存")

    def load_offline_features(self, cid: str) -> Dict:
        """加载离线特征（从缓存或Redis）"""
        if cid in self._offline_cache:
            return self._offline_cache[cid]
        if self._redis:
            data = self._redis.hgetall(f"offline:features:{cid}")
            if data:
                self._offline_cache[cid] = {k: float(v) for k, v in data.items()}
                return self._offline_cache[cid]
        return {}

    def get_prudence_features(self, cid: str) -> Dict:
        """获取适当性特征（稳定，仅离线）"""
        return self.load_offline_features(cid)

    def get_intent_features(self, cid: str, product_id: Optional[str] = None) -> Dict[str, float]:
        """获取意图特征（离线 + 实时融合）"""
        offline = self.load_offline_features(cid)
        realtime = self._realtime_cache.get(cid, {})
        fused = {**offline, **realtime}

        for f in ALL_FEATURE_NAMES:
            if f not in fused:
                fused[f] = 0.0

        for k in list(fused.keys()):
            if isinstance(fused[k], (int, float)):
                fused[k] = float(fused[k])
        return fused

    def ingest_realtime_event(self, event: Dict):
        """摄入实时事件"""
        cid = event.get("customer_id")
        if not cid:
            return
        etype = event.get("event_type")
        val = event.get("event_value", 1.0)

        cache = self._realtime_cache[cid]
        if etype == "calculator_use":
            cache["beh_calculator_use_cnt"] = cache.get("beh_calculator_use_cnt", 0) + val
        elif etype == "compare":
            cache["beh_compare_cnt"] = cache.get("beh_compare_cnt", 0) + val
        elif etype == "view":
            cache["beh_view_cnt_7d"] = cache.get("beh_view_cnt_7d", 0) + val
            cache["beh_view_duration_decay"] = cache.get("beh_view_duration_decay", 0) + val * 5
        elif etype == "revisit":
            cache["beh_revisit_gap_avg"] = 5


# ================================================================
# 编排层
# ================================================================

class NexusOrchestrator:
    """Nexus 编排层"""

    def __init__(
            self,
            data_source: DataSource,
            feature_store: NexusFeatureStore,
            suitability_engine,
            intent_engine,
            decision_engine,
    ):
        self.data_source = data_source
        self.feature_store = feature_store
        self.suitability = suitability_engine
        self.intent = intent_engine
        self.decision = decision_engine
        logger.info("NexusOrchestrator 初始化完成")

    def get_customer_info(self, customer_id: str) -> Dict:
        return self.data_source.get_customer(customer_id)

    def get_product_info(self, product_id: str) -> Dict:
        return self.data_source.get_product(product_id)

    def list_customers(self) -> List[str]:
        return self.data_source.list_customers()

    def list_products(self) -> List[str]:
        return self.data_source.list_products()

    def orchestrate(self, customer_id: str, product_id: str) -> Dict:
        """完整决策编排流程"""
        start_time = time.time()
        logger.info(f"开始编排: {customer_id} -> {product_id}")

        try:
            customer_data = self.get_customer_info(customer_id)
            product_data = self.get_product_info(product_id)

            if not customer_data:
                logger.warning(f"客户不存在: {customer_id}，使用默认数据")
                customer_data = {"risk": "C3", "age": 40, "assets": 100000, "period": 365, "first_buy": False}
            if not product_data:
                logger.warning(f"产品不存在: {product_id}，使用默认数据")
                product_data = {"risk": "R3", "name": "默认产品", "lock": 365, "min": 100000}

            customer = {
                "id": customer_id,
                "risk_level": customer_data.get("risk", "C3"),
                "age": customer_data.get("age", 40),
                "available_assets": customer_data.get("assets", 100000),
                "max_acceptable_period": customer_data.get("period", 365),
                "first_time_buyer": customer_data.get("first_buy", False),
            }

            product = {
                "id": product_id,
                "risk_level": product_data.get("risk", "R3"),
                "name": product_data.get("name", ""),
                "lock_period": product_data.get("lock", 365),
                "min_amount": product_data.get("min", 100000),
            }

            intent_features = self.data_source.get_intent_features(customer_id, product_id)
            if not intent_features:
                intent_features = self.feature_store.get_intent_features(customer_id, product_id)
            customer["intent_features"] = intent_features

            decision = self.decision.decide(customer, product)
            result = decision.to_dict()
            result["customer_name"] = customer_data.get("name", "")
            result["product_name"] = product_data.get("name", "")
            result["latency_ms"] = round((time.time() - start_time) * 1000, 2)

            logger.info(f"编排完成: {result['action']} (耗时 {result['latency_ms']}ms)")
            return result

        except Exception as e:
            logger.error(f"编排失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "customer_id": customer_id,
                "product_id": product_id,
                "action": "ERROR",
                "suitability_level": "UNKNOWN",
                "intent_score": 0.0,
                "reason": f"编排异常: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def batch_orchestrate(self, requests: List[Dict]) -> List[Dict]:
        """批量决策"""
        results = []
        for req in requests:
            cid = req.get("customer_id")
            pid = req.get("product_id")
            if cid and pid:
                results.append(self.orchestrate(cid, pid))
            else:
                results.append({"error": "缺少 customer_id 或 product_id"})
        return results