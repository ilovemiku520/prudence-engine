# intent_subsystem.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pickle
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb

# SHAP 可选
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from loguru import logger

# ================================================================
# 1. 特征注册（保持与之前一致）
# ================================================================
ALL_FEATURE_NAMES = [
    "beh_view_cnt_7d", "beh_view_duration_decay", "beh_calculator_use_cnt",
    "beh_compare_cnt", "beh_revisit_gap_avg", "txn_history_product_types",
    "txn_redemption_freq", "txn_avg_holding_period", "txn_days_since_last_purchase",
    "profile_aum_tier", "profile_lifecycle_stage", "profile_risk_level",
    "interact_push_open_rate_30d", "interact_advisor_contact_freq", "interact_last_script_accepted"
]

# ================================================================
# 2. 规则打分（与之前相同）
# ================================================================
RULE_WEIGHTS = {
    "beh_calculator_use_cnt":   {"weight": 15, "cap": 30},
    "beh_compare_cnt":          {"weight": 10, "cap": 20},
    "beh_view_duration_decay":  {"weight": 8,  "cap": 24},
    "beh_revisit_gap_avg":      {"weight": 12, "cap": 12, "inverse": True},
    "txn_days_since_last_purchase": {"weight": 5, "cap": 10, "inverse": True},
}
INVERSE_THRESHOLD = {"beh_revisit_gap_avg": 72, "txn_days_since_last_purchase": 30}

def compute_rule_score(features: Dict[str, float]) -> float:
    total = 0.0
    for feat, cfg in RULE_WEIGHTS.items():
        raw = features.get(feat, 0)
        if cfg.get("inverse", False):
            th = INVERSE_THRESHOLD.get(feat, 100)
            if raw >= th:
                contrib = 0.0
            else:
                contrib = (1 - raw / th) * cfg["weight"]
            contrib = min(contrib, cfg["cap"])
        else:
            contrib = min(raw * cfg["weight"], cfg["cap"])
        total += contrib
    return min(total / 100.0, 1.0)

# ================================================================
# 3. 意图模型类（封装 XGBoost + SHAP）
# ================================================================
class IntentModel:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.feature_names = ALL_FEATURE_NAMES
        self.is_trained = False
        self._shap_explainer = None
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None, **kwargs):
        X = X[self.feature_names].copy()
        neg = (y == 0).sum()
        pos = (y == 1).sum()
        if pos == 0:
            raise ValueError("正样本为0，无法训练")
        scale = neg / pos
        params = {
            'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'scale_pos_weight': scale, 'eval_metric': 'auc',
            'use_label_encoder': False, 'random_state': 42, 'verbosity': 1
        }
        params.update(kwargs)
        self.model = xgb.XGBClassifier(**params)
        if eval_set is not None:
            self.model.fit(X, y, eval_set=eval_set, verbose=False)
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        self.is_trained = True
        if SHAP_AVAILABLE and self.model is not None:
            try:
                self._shap_explainer = shap.TreeExplainer(
                    self.model, X.sample(min(100, len(X)), random_state=42)
                )
            except:
                pass
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("模型未训练")
        X = X[self.feature_names].copy()
        return self.model.predict_proba(X)

    def predict(self, X: pd.DataFrame, threshold=0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"模型已保存至: {path}")

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        logger.info(f"模型已加载: {path}")

    def explain(self, features: Dict[str, float], top_k=3) -> List[Dict]:
        df = pd.DataFrame([features])[self.feature_names]
        if self._shap_explainer is not None and SHAP_AVAILABLE:
            shap_values = self._shap_explainer.shap_values(df)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            else:
                shap_values = shap_values[0]
            contribs = [
                {"feature": f, "shap_value": float(shap_values[i]),
                 "impact": "positive" if shap_values[i] > 0 else "negative"}
                for i, f in enumerate(self.feature_names)
            ]
            contribs.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            return contribs[:top_k]
        # 降级方案
        return [{"feature": "rule_score", "shap_value": 0.5, "impact": "positive"}]

# ================================================================
# 4. 意图引擎（主逻辑）
# ================================================================
class IntentEngine:
    def __init__(self, model: Optional[IntentModel] = None, model_confidence_stage: str = "stable"):
        self.model = model if model else IntentModel()
        self.stage = model_confidence_stage
        self._feature_cache = {}

    def _get_features(self, customer_id: str) -> Dict[str, float]:
        # 实际对接特征平台，此处模拟（保留原样）
        if customer_id in self._feature_cache:
            return self._feature_cache[customer_id]
        np.random.seed(hash(customer_id) % 2**32)
        feats = {}
        for f in ALL_FEATURE_NAMES:
            if "cnt" in f or "freq" in f:
                feats[f] = max(0, int(np.random.poisson(2)))
            elif "duration" in f:
                feats[f] = max(0, np.random.exponential(30))
            elif "gap" in f or "days" in f:
                feats[f] = max(1, int(np.random.exponential(20)))
            elif "rate" in f:
                feats[f] = round(np.random.beta(2, 5), 2)
            elif "tier" in f or "stage" in f or "level" in f:
                feats[f] = np.random.randint(1, 6)
            else:
                feats[f] = np.random.rand() * 0.5
        self._feature_cache[customer_id] = feats
        return feats

    def get_alpha(self) -> float:
        return {"new": 0.3, "growing": 0.5, "stable": 0.7}.get(self.stage, 0.5)

    def fused_intent_score(self, customer_id: str) -> Dict:
        features = self._get_features(customer_id)
        return self.fused_intent_score_from_features(features)

    def fused_intent_score_from_features(self, features: Dict[str, float]) -> Dict:
        r_score = compute_rule_score(features)
        m_score = 0.0
        if self.model.is_trained:
            try:
                df = pd.DataFrame([features])[self.model.feature_names]
                proba = self.model.predict_proba(df)
                m_score = float(proba[0][1])
            except Exception as e:
                logger.warning(f"模型预测失败，使用规则分替代: {e}")
                m_score = r_score
        else:
            m_score = r_score
        alpha = self.get_alpha()
        final = alpha * m_score + (1 - alpha) * r_score
        final = round(min(max(final, 0.0), 1.0), 4)
        top_signals = []
        if self.model.is_trained:
            top_signals = self.model.explain(features, 3)
        else:
            top_signals = [{"feature": "rule_score", "shap_value": r_score, "impact": "positive"}]
        return {
            "intent_score": final,
            "rule_score": round(r_score, 4),
            "model_score": round(m_score, 4),
            "top_signals": top_signals
        }

# ================================================================
# 5. 训练管道（供 main.py 调用）
# ================================================================
class IntentTrainPipeline:
    """意图模型训练管道（用于生成合成数据或真实数据训练）"""

    @staticmethod
    def generate_synthetic_data(n_customers: int = 2000, seed: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """生成模拟数据用于演示训练"""
        np.random.seed(seed)
        customers = [f"CUST_{i:05d}" for i in range(n_customers)]
        rows = []
        labels = []
        for cid in customers:
            row = {}
            for meta in ALL_FEATURE_NAMES:
                name = meta
                if "cnt" in name:
                    row[name] = np.random.poisson(1.5)
                elif "duration" in name:
                    row[name] = max(0, np.random.exponential(20))
                elif "gap" in name or "days" in name:
                    row[name] = max(1, int(np.random.exponential(15)))
                elif "rate" in name:
                    row[name] = round(np.random.beta(1.5, 4), 2)
                elif "tier" in name or "stage" in name or "level" in name:
                    row[name] = np.random.randint(1, 6)
                else:
                    row[name] = np.random.rand()
            # 高意向样本：计算器使用>2或比较>3或浏览时长>50
            is_high = (row.get("beh_calculator_use_cnt", 0) > 2 or
                       row.get("beh_compare_cnt", 0) > 3 or
                       row.get("beh_view_duration_decay", 0) > 50)
            label = 1 if (is_high and np.random.rand() < 0.4) else (1 if np.random.rand() < 0.05 else 0)
            rows.append(row)
            labels.append(label)
        X = pd.DataFrame(rows)[ALL_FEATURE_NAMES]
        y = pd.Series(labels)
        print(f"[合成数据] 共 {n_customers} 条，正样本率: {y.mean():.2%}")
        return X, y

    @classmethod
    def run_training(cls, X: pd.DataFrame, y: pd.Series, save_path: str = "intent_model.pkl") -> IntentModel:
        """执行完整训练流程"""
        logger.info(f"开始训练意图模型，样本数: {len(X)}，正样本率: {y.mean():.4f}")
        model = IntentModel()
        model.fit(X, y)
        # 评估
        from sklearn.metrics import roc_auc_score
        proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)
        logger.info(f"训练集 AUC: {auc:.4f}")
        model.save(save_path)
        return model