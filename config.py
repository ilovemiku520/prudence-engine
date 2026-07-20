# config.py
"""
配置管理模块 - 支持环境变量 / 配置文件 / 命令行参数
"""
import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path


# ================================================================
# 常量定义（统一管理硬编码值）
# ================================================================

class Defaults:
    """默认配置常量"""
    # 数据源
    DS_TYPE = "mock"
    DS_HOST = "localhost"
    DS_PORT = 3306
    DS_DATABASE = "prudence"
    DS_USER = "root"
    DS_PASSWORD = ""
    DS_DB_PATH = "./prudence.db"

    # 意图引擎
    INTENT_MODEL_PATH = "./intent_model.pkl"
    INTENT_STAGE = "stable"
    INTENT_THRESHOLD_HIGH = 0.7
    INTENT_THRESHOLD_LOW = 0.4
    INTENT_MAX_TRAINING = 5000

    # 特征存储
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 0
    REDIS_ENABLE = False
    CACHE_TTL = 3600

    # API
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_DEBUG = False

    # 日志
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "json"
    LOG_DIR = "./logs"

    # 矩阵
    MATRIX_VERSION = "v1.0"


@dataclass
class DataSourceConfig:
    """数据源配置"""
    type: str = Defaults.DS_TYPE
    host: str = Defaults.DS_HOST
    port: int = Defaults.DS_PORT
    database: str = Defaults.DS_DATABASE
    user: str = Defaults.DS_USER
    password: str = Defaults.DS_PASSWORD
    db_path: str = Defaults.DS_DB_PATH

    def to_dict(self) -> Dict:
        return {
            "type": self.type,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password,
            "db_path": self.db_path,
        }


@dataclass
class IntentConfig:
    """意图引擎配置"""
    model_path: str = Defaults.INTENT_MODEL_PATH
    confidence_stage: str = Defaults.INTENT_STAGE
    threshold_high: float = Defaults.INTENT_THRESHOLD_HIGH
    threshold_low: float = Defaults.INTENT_THRESHOLD_LOW
    max_training_samples: int = Defaults.INTENT_MAX_TRAINING


@dataclass
class SuitabilityConfig:
    """适当性引擎配置"""
    matrix_version: str = Defaults.MATRIX_VERSION
    enable_rules: bool = True
    enable_financial_check: bool = True
    enable_period_check: bool = True
@dataclass
class FeatureStoreConfig:
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    enable_redis: bool = True  # 改为 True
    cache_ttl: int = 3600


@dataclass
class APIConfig:
    """API服务配置"""
    host: str = Defaults.API_HOST
    port: int = Defaults.API_PORT
    debug: bool = Defaults.API_DEBUG
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_per_minute: int = 60


@dataclass
class LoggerConfig:
    """日志配置"""
    level: str = Defaults.LOG_LEVEL
    format: str = Defaults.LOG_FORMAT
    output_dir: str = Defaults.LOG_DIR
    max_size_mb: int = 100
    retention_days: int = 30
    enable_audit: bool = True


@dataclass
class AppConfig:
    """应用总配置"""
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    intent: IntentConfig = field(default_factory=IntentConfig)
    suitability: SuitabilityConfig = field(default_factory=SuitabilityConfig)
    feature_store: FeatureStoreConfig = field(default_factory=FeatureStoreConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)

    enable_auth: bool = True
    report_output_dir: str = "./reports"

    @classmethod
    def from_env(cls) -> "AppConfig":
        """从环境变量加载配置"""
        config = cls()
        ds = config.data_source
        ds.type = os.getenv("DS_TYPE", Defaults.DS_TYPE)
        ds.host = os.getenv("DS_HOST", Defaults.DS_HOST)
        ds.port = int(os.getenv("DS_PORT", str(Defaults.DS_PORT)))
        ds.database = os.getenv("DS_DATABASE", Defaults.DS_DATABASE)
        ds.user = os.getenv("DS_USER", Defaults.DS_USER)
        ds.password = os.getenv("DS_PASSWORD", Defaults.DS_PASSWORD)
        ds.db_path = os.getenv("DS_DB_PATH", Defaults.DS_DB_PATH)

        intent = config.intent
        intent.model_path = os.getenv("INTENT_MODEL_PATH", Defaults.INTENT_MODEL_PATH)
        intent.confidence_stage = os.getenv("INTENT_STAGE", Defaults.INTENT_STAGE)
        intent.threshold_high = float(os.getenv("INTENT_THRESHOLD_HIGH", str(Defaults.INTENT_THRESHOLD_HIGH)))
        intent.threshold_low = float(os.getenv("INTENT_THRESHOLD_LOW", str(Defaults.INTENT_THRESHOLD_LOW)))

        api = config.api
        api.host = os.getenv("API_HOST", Defaults.API_HOST)
        api.port = int(os.getenv("API_PORT", str(Defaults.API_PORT)))
        api.debug = os.getenv("API_DEBUG", "false").lower() == "true"

        logger = config.logger
        logger.level = os.getenv("LOG_LEVEL", Defaults.LOG_LEVEL)
        logger.format = os.getenv("LOG_FORMAT", Defaults.LOG_FORMAT)
        logger.output_dir = os.getenv("LOG_DIR", Defaults.LOG_DIR)

        return config

    @classmethod
    def from_file(cls, path: str) -> "AppConfig":
        """从JSON配置文件加载"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        config = cls()
        if "data_source" in data:
            for key, val in data["data_source"].items():
                if hasattr(config.data_source, key):
                    setattr(config.data_source, key, val)
        if "intent" in data:
            for key, val in data["intent"].items():
                if hasattr(config.intent, key):
                    setattr(config.intent, key, val)
        if "api" in data:
            for key, val in data["api"].items():
                if hasattr(config.api, key):
                    setattr(config.api, key, val)
        if "logger" in data:
            for key, val in data["logger"].items():
                if hasattr(config.logger, key):
                    setattr(config.logger, key, val)
        return config

    def save(self, path: str):
        """保存配置到JSON文件"""
        data = {
            "data_source": self.data_source.to_dict(),
            "intent": {
                "model_path": self.intent.model_path,
                "confidence_stage": self.intent.confidence_stage,
                "threshold_high": self.intent.threshold_high,
                "threshold_low": self.intent.threshold_low,
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "debug": self.api.debug,
            },
            "logger": {
                "level": self.logger.level,
                "format": self.logger.format,
                "output_dir": self.logger.output_dir,
            }
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


_global_config: Optional[AppConfig] = None


def get_config(reload: bool = False) -> AppConfig:
    """获取全局配置（单例模式）"""
    global _global_config
    if _global_config is None or reload:
        _global_config = AppConfig.from_env()
    return _global_config


def set_config(config: AppConfig):
    """设置全局配置"""
    global _global_config
    _global_config = config