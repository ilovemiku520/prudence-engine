# main.py
"""
睿衡引擎 - 主入口
支持：决策 / 批量决策 / 评估 / API服务 / UI启动
支持：自定义数据源注入（用于文件上传等场景）
"""
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# 导入所有模块
from config import get_config, AppConfig, Defaults
from logger import get_logger, StructuredLogger
from data_source import create_data_source, MockDataSource, DataSourceError, DataSource
from prudence_suitability import SuitabilityMatrixConfig, SuitabilityEngine
from intent_subsystem import IntentModel, IntentEngine, IntentTrainPipeline
from aegis_decision import AegisDecisionEngine
from nexus_orchestrator import NexusFeatureStore, NexusOrchestrator

# 评估模块（可选）
try:
    from validity_evaluator import ValidityEvaluator, EvaluationDataGenerator
except ImportError:
    ValidityEvaluator = None


# ================================================================
# 1. 引擎工厂
# ================================================================

class PrudenceFactory:
    """Prudence 引擎工厂 - 负责创建所有组件实例"""

    @staticmethod
    def create_engines(config: AppConfig, data_source: Optional[DataSource] = None) -> Dict:
        """
        创建并组装所有引擎

        参数：
            config: 应用配置
            data_source: 可选的自定义数据源（如 MemoryDataSource），
                        若为 None，则根据 config 自动创建

        返回：包含所有引擎和组件的字典
        """
        logger = get_logger()
        logger.info("开始创建引擎组件...")

        # ---- 1. 创建数据源 ----
        if data_source is None:
            try:
                ds_config = config.data_source
                data_source = create_data_source(
                    ds_config.type,
                    host=ds_config.host,
                    port=ds_config.port,
                    database=ds_config.database,
                    user=ds_config.user,
                    password=ds_config.password,
                    db_path=ds_config.db_path
                )
                logger.info(f"数据源已创建: {ds_config.type}")
            except DataSourceError as e:
                logger.error(f"数据源创建失败: {e}")
                raise
            except Exception as e:
                logger.error(f"数据源创建异常: {e}")
                raise
        else:
            logger.info("使用自定义数据源（如文件上传）")

        # ---- 2. 加载产品数据库 ----
        product_db = []
        try:
            for pid in data_source.list_products():
                prod = data_source.get_product(pid)
                if prod:
                    product_db.append({
                        "id": pid,
                        "risk_level": prod.get("risk", "R1"),
                        "name": prod.get("name", ""),
                        "lock_period": prod.get("lock", 0),
                        "min_amount": prod.get("min", 0),
                    })
            logger.info(f"从数据源加载了 {len(product_db)} 个产品")
        except Exception as e:
            logger.warning(f"加载产品失败: {e}，使用默认产品库")

        # 如果数据源没有产品，使用默认产品库
        if not product_db:
            logger.warning("数据源无产品，使用默认产品库")
            product_db = [
                {"id": "P001", "name": "天天利货币", "risk_level": "R1", "lock_period": 0, "min_amount": 0},
                {"id": "P002", "name": "季季盈固收", "risk_level": "R2", "lock_period": 90, "min_amount": 10000},
                {"id": "P004", "name": "一年平衡混合", "risk_level": "R3", "lock_period": 365, "min_amount": 100000},
                {"id": "P005", "name": "进取科创主题", "risk_level": "R4", "lock_period": 730, "min_amount": 200000},
                {"id": "P006", "name": "全球精选权益", "risk_level": "R5", "lock_period": 1095, "min_amount": 500000},
            ]

        # ---- 3. 适当性引擎 ----
        matrix = SuitabilityMatrixConfig()
        suitability_engine = SuitabilityEngine(matrix, product_db)
        logger.info("适当性引擎已创建")

        # ---- 4. 意图引擎 ----
        intent_model_path = config.intent.model_path
        intent_model = IntentModel(intent_model_path)

        # 如果模型不存在，自动训练
        if not intent_model.is_trained:
            logger.info("意图模型不存在，开始训练...")
            X, y = IntentTrainPipeline.generate_synthetic_data(config.intent.max_training_samples)
            intent_model.fit(X, y)
            intent_model.save(intent_model_path)
            logger.info(f"意图模型训练完成，保存至: {intent_model_path}")

        intent_engine = IntentEngine(intent_model, config.intent.confidence_stage)
        logger.info("意图引擎已创建")

        # ---- 5. 特征存储 ----
        feature_store = NexusFeatureStore(
            config.feature_store.redis_host,
            config.feature_store.redis_port
        )
        logger.info("特征存储已创建")

        # ---- 6. 融合决策引擎 ----
        decision_engine = AegisDecisionEngine(
            suitability_engine,
            intent_engine,
            {
                "intent_threshold_high": config.intent.threshold_high,
                "intent_threshold_low": config.intent.threshold_low,
            }
        )
        logger.info("融合决策引擎已创建")

        # ---- 7. 编排层 ----
        orchestrator = NexusOrchestrator(
            data_source=data_source,
            feature_store=feature_store,
            suitability_engine=suitability_engine,
            intent_engine=intent_engine,
            decision_engine=decision_engine,
        )
        logger.info("编排层已创建")

        return {
            "data_source": data_source,
            "suitability": suitability_engine,
            "intent": intent_engine,
            "feature_store": feature_store,
            "decision": decision_engine,
            "orchestrator": orchestrator,
        }


# ================================================================
# 2. Prudence API（高层接口）
# ================================================================

class PrudenceAPI:
    """
    Prudence 高层 API - 对外提供统一接口

    使用示例：
        # 使用默认配置和数据源
        api = PrudenceAPI()
        result = api.decide("CUST_HIGH", "P004")

        # 使用自定义数据源（如文件上传）
        from data_source import MemoryDataSource
        ds = MemoryDataSource(customers=my_customers, products=my_products)
        api = PrudenceAPI(data_source=ds)
        result = api.decide("CUST_001", "P001")
    """

    def __init__(self, config: Optional[AppConfig] = None, data_source: Optional[DataSource] = None):
        """
        初始化 API

        参数：
            config: 应用配置，若为 None 则自动加载
            data_source: 可选的自定义数据源，若提供则优先使用
        """
        self.config = config or get_config()
        self.engines = PrudenceFactory.create_engines(self.config, data_source=data_source)
        self.orchestrator = self.engines["orchestrator"]
        self.data_source = self.engines["data_source"]
        self.logger = get_logger()
        self.logger.info("PrudenceAPI 初始化完成")

    def decide(self, customer_id: str, product_id: str) -> Dict:
        """单次决策"""
        return self.orchestrator.orchestrate(customer_id, product_id)

    def batch_decide(self, requests: List[Dict]) -> List[Dict]:
        """批量决策"""
        return self.orchestrator.batch_orchestrate(requests)

    def list_customers(self) -> List[str]:
        """获取客户列表"""
        return self.orchestrator.list_customers()

    def list_products(self) -> List[str]:
        """获取产品列表"""
        return self.orchestrator.list_products()

    def get_customer(self, customer_id: str) -> Dict:
        """获取客户详情"""
        return self.orchestrator.get_customer_info(customer_id)

    def get_product(self, product_id: str) -> Dict:
        """获取产品详情"""
        return self.orchestrator.get_product_info(product_id)

    def run_offline_job(self, date: Optional[str] = None) -> Dict:
        """运行离线特征计算（模拟）"""
        logger = get_logger()
        target_date = date or datetime.now().strftime('%Y-%m-%d')
        logger.info(f"运行离线特征计算: {target_date}")
        return {"status": "completed", "date": target_date}

    def generate_eval_report(self, report_name: str = "validity_report") -> Dict:
        """生成评估报告"""
        if ValidityEvaluator is None:
            return {"error": "评估模块未安装 (validity_evaluator.py)"}

        gen = EvaluationDataGenerator()
        suit = gen.generate_suitability_samples(1500)
        io = gen.generate_intent_samples(3000, False)
        ion = gen.generate_intent_samples(2000, True)
        ctrl = gen.generate_intent_samples(1000, True)
        fusion = gen.generate_fusion_samples(800)

        evaluator = ValidityEvaluator(self.config.report_output_dir)
        return evaluator.generate_full_report(suit, io, ion, fusion, ctrl, report_name)

    def get_data_source_info(self) -> Dict:
        """获取当前数据源信息（用于调试/UI展示）"""
        ds = self.data_source
        ds_type = type(ds).__name__
        return {
            "type": ds_type,
            "customers_count": len(ds.list_customers()),
            "products_count": len(ds.list_products()),
        }


# ================================================================
# 3. 命令行入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="睿衡引擎 - 统一入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 单次决策
  python main.py --mode decision --customer CUST_HIGH --product P004

  # 批量决策
  echo '[{"customer_id":"CUST_HIGH","product_id":"P004"}]' | python main.py --mode batch

  # 启动 API 服务
  python main.py --mode api

  # 启动 UI 仪表板
  python main.py --mode ui --port 8501

  # 生成评估报告
  python main.py --mode eval --report my_report

  # 使用配置文件
  python main.py --mode api --config config.json
        """
    )
    parser.add_argument(
        "--mode",
        choices=["decision", "batch", "offline", "eval", "api", "ui"],
        default="decision",
        help="运行模式"
    )
    parser.add_argument("--customer", help="客户ID (decision模式)")
    parser.add_argument("--product", help="产品ID (decision模式)")
    parser.add_argument("--config", help="配置文件路径 (JSON)")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="离线计算日期")
    parser.add_argument("--report", default="validity_report", help="评估报告名称")
    parser.add_argument("--port", type=int, default=8501, help="UI端口 (ui模式)")

    args = parser.parse_args()

    # 加载配置
    if args.config and Path(args.config).exists():
        config = AppConfig.from_file(args.config)
    else:
        config = get_config()

    # 初始化日志
    logger = StructuredLogger(
        name="prudence",
        level=config.logger.level,
        log_dir=config.logger.output_dir
    )
    logger.info(f"睿衡引擎启动 | 模式: {args.mode}")

    # 创建 API 实例（使用配置，不传入自定义数据源）
    api = PrudenceAPI(config)

    # 执行
    if args.mode == "decision":
        if not args.customer or not args.product:
            print("错误: decision模式需要 --customer 和 --product")
            sys.exit(1)
        result = api.decide(args.customer, args.product)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.mode == "batch":
        data = sys.stdin.read()
        requests = json.loads(data) if data else [{"customer_id": "CUST_HIGH", "product_id": "P004"}]
        results = api.batch_decide(requests)
        print(json.dumps(results, indent=2, ensure_ascii=False))

    elif args.mode == "offline":
        result = api.run_offline_job(args.date)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.mode == "eval":
        report = api.generate_eval_report(args.report)
        print(f"评估报告已生成: {config.report_output_dir}/{args.report}.json")

    elif args.mode == "api":
        from api import create_app
        import uvicorn
        app = create_app(config)
        logger.info(f"启动 API 服务: http://{config.api.host}:{config.api.port}")
        logger.info(f"API 文档: http://{config.api.host}:{config.api.port}/api/docs")
        uvicorn.run(
            app,
            host=config.api.host,
            port=config.api.port,
            log_level=config.logger.level.lower(),
            access_log=config.api.debug,
        )

    elif args.mode == "ui":
        # 启动 Streamlit UI
        import subprocess
        import os
        ui_path = Path(__file__).parent / "ui.py"
        if not ui_path.exists():
            print(f"错误: ui.py 不存在于 {ui_path}")
            sys.exit(1)
        logger.info(f"启动 UI 仪表板: http://localhost:{args.port}")
        subprocess.run(["streamlit", "run", str(ui_path), "--server.port", str(args.port)])

    else:
        print("未知模式")
        sys.exit(1)


if __name__ == "__main__":
    main()