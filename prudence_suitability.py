# prudence_suitability.py
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field


class MatchLevel(Enum):
    ALLOW = 1
    RESTRICTED = 2
    FORBID = 3

    @classmethod
    def tighten(cls, current: 'MatchLevel', rule: 'MatchLevel') -> 'MatchLevel':
        return rule if rule.value > current.value else current

    @classmethod
    def tighten_to_at_least(cls, current: 'MatchLevel', threshold: 'MatchLevel') -> 'MatchLevel':
        return threshold if threshold.value > current.value else current

    @classmethod
    def from_string(cls, s: str) -> 'MatchLevel':
        mapping = {"ALLOW": cls.ALLOW, "RESTRICTED": cls.RESTRICTED, "FORBID": cls.FORBID}
        return mapping.get(s, cls.FORBID)


@dataclass
class SuitabilityResult:
    level: str
    matched_rule: str
    restriction_reason: Optional[str] = None
    replacement_products: List[Dict] = field(default_factory=list)
    special_rules_triggered: List[str] = field(default_factory=list)


# ================================================================
# 规则类（提取到外部，避免重复创建）
# ================================================================

class SuitabilityRule:
    """适当性规则"""
    def __init__(self, name: str, desc: str, func: Callable[[Dict, Dict], Optional[MatchLevel]]):
        self.name = name
        self.desc = desc
        self.func = func

    def execute(self, customer: Dict, product: Dict) -> tuple:
        return self.func(customer, product), self.desc


class SuitabilityMatrixConfig:
    def __init__(self):
        self._matrix = self._default_matrix()

    def _default_matrix(self) -> Dict[str, Dict[str, str]]:
        return {
            "C1": {"R1": "ALLOW", "R2": "FORBID", "R3": "FORBID", "R4": "FORBID", "R5": "FORBID"},
            "C2": {"R1": "ALLOW", "R2": "ALLOW", "R3": "FORBID", "R4": "FORBID", "R5": "FORBID"},
            "C3": {"R1": "ALLOW", "R2": "ALLOW", "R3": "ALLOW", "R4": "FORBID", "R5": "FORBID"},
            "C4": {"R1": "ALLOW", "R2": "ALLOW", "R3": "ALLOW", "R4": "ALLOW", "R5": "RESTRICTED"},
            "C5": {"R1": "ALLOW", "R2": "ALLOW", "R3": "ALLOW", "R4": "ALLOW", "R5": "ALLOW"},
        }

    def get_cell(self, c_level: str, r_level: str) -> str:
        return self._matrix.get(c_level, {}).get(r_level, "FORBID")


class SuitabilityEngine:
    def __init__(self, matrix_config: SuitabilityMatrixConfig, product_db: List[Dict]):
        self.matrix_config = matrix_config
        self.product_db = {p["id"]: p for p in product_db}
        self.special_rules = self._build_rules()

    @staticmethod
    def _build_rules() -> List[SuitabilityRule]:
        """构建规则列表（静态方法，只构建一次）"""
        rules = []

        # 高龄65+ R3+
        rules.append(SuitabilityRule(
            "HIGH_AGE_65", "高龄(≥65岁)购买R3+需确认",
            lambda c, p: MatchLevel.RESTRICTED if (c.get("age", 0) >= 65 and int(p.get("risk_level", "R1")[1]) >= 3) else None
        ))

        # 首次购买R4+禁止
        rules.append(SuitabilityRule(
            "FIRST_TIME_R4", "首次购买R4+禁止",
            lambda c, p: MatchLevel.FORBID if (c.get("first_time_buyer", False) and int(p.get("risk_level", "R1")[1]) >= 4) else None
        ))

        # 财务匹配
        def fin_rule(c, p):
            amt = p.get("min_amount", 0)
            assets = c.get("available_assets", 0)
            if assets <= 0:
                return MatchLevel.RESTRICTED
            ratio = amt / assets
            if ratio > 0.7:
                return MatchLevel.FORBID
            if ratio > 0.4:
                return MatchLevel.RESTRICTED
            return None
        rules.append(SuitabilityRule("FIN_CAP", "财务匹配", fin_rule))

        # 期限匹配
        def period_rule(c, p):
            lock = p.get("lock_period", 0)
            maxp = c.get("max_acceptable_period", 365)
            if lock > maxp * 2:
                return MatchLevel.FORBID
            if lock > maxp:
                return MatchLevel.RESTRICTED
            return None
        rules.append(SuitabilityRule("PERIOD", "期限匹配", period_rule))

        return rules

    def _evaluate_base(self, customer: Dict, product: Dict) -> SuitabilityResult:
        """内部判定函数，不做替代推荐（避免递归）"""
        c_level = customer.get("risk_level", "C1")
        r_level = product.get("risk_level", "R1")
        cell = self.matrix_config.get_cell(c_level, r_level)
        base_level = MatchLevel.from_string(cell)
        current = base_level
        triggered = []

        for rule in self.special_rules:
            new_level, desc = rule.execute(customer, product)
            if new_level is not None and new_level.value > current.value:
                current = new_level
                triggered.append(rule.name)

        level_map = {MatchLevel.ALLOW: "ALLOW", MatchLevel.RESTRICTED: "RESTRICTED", MatchLevel.FORBID: "FORBID"}
        level_str = level_map[current]

        reason = None
        if level_str == "FORBID":
            reason = f"产品{product.get('id', '')}不满足适当性"
        elif level_str == "RESTRICTED":
            reason = f"产品{product.get('id', '')}需二次确认"

        return SuitabilityResult(
            level=level_str,
            matched_rule=f"{c_level}x{r_level}",
            restriction_reason=reason,
            replacement_products=[],
            special_rules_triggered=triggered
        )

    def check_suitability(self, customer: Dict, product: Dict) -> SuitabilityResult:
        result = self._evaluate_base(customer, product)
        if result.level == "FORBID":
            replacements = self.find_alternatives(customer, product.get("id"))
            result.replacement_products = replacements
        return result

    def find_alternatives(self, customer: Dict, exclude_id: Optional[str] = None, top_k: int = 5) -> List[Dict]:
        allowed = []
        for pid, prod in self.product_db.items():
            if pid == exclude_id:
                continue
            result = self._evaluate_base(customer, prod)
            if result.level in ["ALLOW", "RESTRICTED"]:
                allowed.append(prod)
        allowed.sort(key=lambda x: (int(x.get("risk_level", "R1")[1]), x.get("lock_period", 0)))
        return allowed[:top_k]