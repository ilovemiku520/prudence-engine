# logger.py
"""
日志与审计追踪模块 - 结构化日志 + 合规审计
"""
import json
import sys
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import uuid

# 全局审计日志存储（可替换为数据库）
_audit_logs: List[Dict] = []


class StructuredLogger:
    """结构化日志记录器（支持JSON格式）"""

    def __init__(self, name: str = "prudence", level: str = "INFO", log_dir: str = "./logs"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_handlers()

    def _setup_handlers(self):
        # 控制台输出
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.logger.addHandler(console_handler)

        # 文件输出（按大小轮转）
        file_handler = RotatingFileHandler(
            self.log_dir / f"{self.name}.log",
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=10,
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.logger.addHandler(file_handler)

        # 审计日志单独文件
        audit_handler = RotatingFileHandler(
            self.log_dir / f"{self.name}_audit.log",
            maxBytes=100 * 1024 * 1024,
            backupCount=20,
            encoding='utf-8'
        )
        audit_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.logger.addHandler(audit_handler)

    def _format_message(self, level: str, message: str, extra: Optional[Dict] = None) -> str:
        if extra:
            extra_str = " | " + " | ".join(f"{k}={v}" for k, v in extra.items())
        else:
            extra_str = ""
        return f"{level} | {message}{extra_str}"

    def info(self, message: str, **kwargs):
        self.logger.info(self._format_message("INFO", message, kwargs))

    def warning(self, message: str, **kwargs):
        self.logger.warning(self._format_message("WARNING", message, kwargs))

    def error(self, message: str, **kwargs):
        self.logger.error(self._format_message("ERROR", message, kwargs))

    def debug(self, message: str, **kwargs):
        self.logger.debug(self._format_message("DEBUG", message, kwargs))

    def audit(self, event: str, customer_id: str = "", product_id: str = "",
              trace_id: Optional[str] = None, details: Optional[Dict] = None) -> str:
        """
        审计日志（合规追踪）
        记录所有关键决策节点，满足监管回溯要求
        """
        if trace_id is None:
            trace_id = str(uuid.uuid4())

        audit_record = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "customer_id": customer_id,
            "product_id": product_id,
            "trace_id": trace_id,
            "details": details or {}
        }
        # 写入审计日志文件
        self.logger.info(f"AUDIT | {json.dumps(audit_record, ensure_ascii=False)}")
        # 同时存入内存（供UI查看）
        _audit_logs.append(audit_record)
        # 限制内存大小
        if len(_audit_logs) > 1000:
            _audit_logs.pop(0)
        return trace_id


# 全局日志实例
_logger: Optional[StructuredLogger] = None


def get_logger() -> StructuredLogger:
    """获取全局日志实例"""
    global _logger
    if _logger is None:
        _logger = StructuredLogger()
    return _logger


def get_audit_logs(limit: int = 100) -> List[Dict]:
    """获取审计日志（用于UI展示）"""
    return _audit_logs[-limit:] if _audit_logs else []


def clear_audit_logs():
    """清空审计日志（仅内存中的）"""
    global _audit_logs
    _audit_logs = []


class MetricsCollector:
    """指标收集器（用于监控）"""

    def __init__(self):
        self._metrics = {
            "decision_total": 0,
            "decision_block": 0,
            "decision_review": 0,
            "decision_close": 0,
            "decision_nurture": 0,
            "intent_score_sum": 0.0,
            "latency_sum_ms": 0.0,
        }
        self._counters = {
            "decision_total": 0,
            "decision_block": 0,
            "decision_review": 0,
            "decision_close": 0,
            "decision_nurture": 0,
        }
        self._reset_counters()

    def _reset_counters(self):
        self._counters = {
            "decision_total": 0,
            "decision_block": 0,
            "decision_review": 0,
            "decision_close": 0,
            "decision_nurture": 0,
        }

    def record_decision(self, result: Dict, latency_ms: float = 0.0):
        """记录决策结果"""
        action = result.get("action", "UNKNOWN")
        self._counters["decision_total"] += 1
        if "BLOCK" in action:
            self._counters["decision_block"] += 1
        elif "REVIEW" in action:
            self._counters["decision_review"] += 1
        elif "CLOSING" in action:
            self._counters["decision_close"] += 1
        elif "NURTURE" in action:
            self._counters["decision_nurture"] += 1

        self._metrics["intent_score_sum"] += result.get("intent_score", 0)
        self._metrics["latency_sum_ms"] += latency_ms

    def get_metrics(self, reset: bool = False) -> Dict:
        """获取指标汇总"""
        total = self._counters["decision_total"]
        metrics = {
            "total_decisions": total,
            "block_count": self._counters["decision_block"],
            "review_count": self._counters["decision_review"],
            "close_count": self._counters["decision_close"],
            "nurture_count": self._counters["decision_nurture"],
            "avg_intent_score": self._metrics["intent_score_sum"] / total if total > 0 else 0,
            "latency_avg_ms": self._metrics["latency_sum_ms"] / total if total > 0 else 0,
        }
        if reset:
            self._reset_counters()
            self._metrics["intent_score_sum"] = 0.0
            self._metrics["latency_sum_ms"] = 0.0
        return metrics


# 全局指标收集器
_metrics = MetricsCollector()


def get_metrics() -> Dict:
    return _metrics.get_metrics()


def record_decision(result: Dict, latency_ms: float = 0.0):
    _metrics.record_decision(result, latency_ms)