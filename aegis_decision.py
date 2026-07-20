# aegis_decision.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
from loguru import logger

class DecisionAction(Enum):
    BLOCK_AND_REPLACE = "BLOCK_AND_REPLACE"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
    PROACTIVE_CLOSING = "PROACTIVE_CLOSING"
    NURTURE_CONTENT = "NURTURE_CONTENT"
    LOW_PRIORITY = "LOW_PRIORITY"

@dataclass
class UnifiedDecision:
    customer_id: str
    product_id: str
    action: DecisionAction
    suitability_level: str
    intent_score: float = 0.0
    rule_score: float = 0.0
    model_score: float = 0.0
    top_signals: List[Dict] = field(default_factory=list)
    replacement_products: List[Dict] = field(default_factory=list)
    reason: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "customer_id": self.customer_id,
            "product_id": self.product_id,
            "action": self.action.value,
            "suitability_level": self.suitability_level,
            "intent_score": self.intent_score,
            "rule_score": self.rule_score,
            "model_score": self.model_score,
            "top_signals": self.top_signals,
            "replacement_products": self.replacement_products,
            "reason": self.reason,
            "timestamp": self.timestamp
        }


class AegisDecisionEngine:
    def __init__(self, suitability_engine, intent_engine, config: Optional[Dict] = None):
        self.suitability_engine = suitability_engine
        self.intent_engine = intent_engine
        self.config = config or {"intent_threshold_high":0.7, "intent_threshold_low":0.4}

    def decide(self, customer: Dict, product: Dict) -> UnifiedDecision:
        cid = customer.get("id", "unknown")
        pid = product.get("id", "unknown")
        logger.info(f"Aegis 决策: {cid} -> {pid}")

        # 1. 适当性
        suit = self.suitability_engine.check_suitability(customer, product)
        level = suit.level
        rule = suit.matched_rule

        # 2. 若FORBID
        if level == "FORBID":
            return UnifiedDecision(
                customer_id=cid, product_id=pid,
                action=DecisionAction.BLOCK_AND_REPLACE,
                suitability_level=level,
                replacement_products=suit.replacement_products,
                reason=f"适当性不通过({rule})"
            )

        # 3. 意图计算（非FORBID时）
        # 需要从customer中提取特征，这里假设customer已包含特征（由上层注入）
        # 如果customer中没有特征，则从意图引擎的默认获取（但由Nexus提供）
        # 这里我们假定customer中有"intent_features"字段
        features = customer.get("intent_features", {})
        intent_result = self.intent_engine.fused_intent_score_from_features(features)

        # 4. 若RESTRICTED
        if level == "RESTRICTED":
            return UnifiedDecision(
                customer_id=cid, product_id=pid,
                action=DecisionAction.HUMAN_REVIEW_REQUIRED,
                suitability_level=level,
                intent_score=intent_result["intent_score"],
                rule_score=intent_result["rule_score"],
                model_score=intent_result["model_score"],
                top_signals=intent_result["top_signals"],
                reason="适当性受限，需人工双录"
            )

        # 5. ALLOW，按意图分分层
        score = intent_result["intent_score"]
        high = self.config["intent_threshold_high"]
        low = self.config["intent_threshold_low"]
        if score >= high:
            action = DecisionAction.PROACTIVE_CLOSING
            reason = "高意向，促单"
        elif score >= low:
            action = DecisionAction.NURTURE_CONTENT
            reason = "中等意向，培育"
        else:
            action = DecisionAction.LOW_PRIORITY
            reason = "低意向，暂不打扰"

        return UnifiedDecision(
            customer_id=cid, product_id=pid,
            action=action,
            suitability_level=level,
            intent_score=score,
            rule_score=intent_result["rule_score"],
            model_score=intent_result["model_score"],
            top_signals=intent_result["top_signals"],
            reason=reason
        )