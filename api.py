# api.py
"""
API 服务层 - 提供 RESTful 接口
"""
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import time
from datetime import datetime

# 导入各模块
from main import PrudenceAPI
from config import get_config, AppConfig
from logger import get_logger, record_decision


# ================================================================
# 1. Pydantic 模型定义
# ================================================================

class DecisionRequest(BaseModel):
    """决策请求"""
    customer_id: str = Field(..., description="客户ID")
    product_id: str = Field(..., description="产品ID")


class BatchDecisionRequest(BaseModel):
    """批量决策请求"""
    requests: List[DecisionRequest] = Field(..., description="决策请求列表")


class TopSignal(BaseModel):
    """Top信号"""
    feature: str
    shap_value: float
    impact: str


class ReplacementProduct(BaseModel):
    """替代推荐产品"""
    id: str
    name: str
    risk_level: str
    lock_period: int
    min_amount: int


class DecisionResponse(BaseModel):
    """决策响应"""
    customer_id: str
    product_id: str
    action: str
    suitability_level: str
    intent_score: float
    rule_score: float = 0.0
    model_score: float = 0.0
    top_signals: List[TopSignal] = []
    replacement_products: List[ReplacementProduct] = []
    reason: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class BatchDecisionResponse(BaseModel):
    """批量决策响应"""
    results: List[DecisionResponse]
    total: int
    latency_ms: float


class MetricsResponse(BaseModel):
    """指标响应"""
    total_decisions: int
    block_count: int
    review_count: int
    close_count: int
    nurture_count: int
    avg_intent_score: float
    latency_avg_ms: float


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    timestamp: str


# ================================================================
# 2. 应用初始化
# ================================================================

def create_app(config: Optional[AppConfig] = None) -> FastAPI:
    """创建 FastAPI 应用"""
    if config is None:
        config = get_config()

    app = FastAPI(
        title="睿衡引擎 API",
        description="适当性与意图联合决策引擎",
        version="2.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ================================================================
    # 3. 依赖注入
    # ================================================================

    _api_instance = None

    def get_prudence_api() -> PrudenceAPI:
        nonlocal _api_instance
        if _api_instance is None:
            _api_instance = PrudenceAPI(config)
        return _api_instance

    # ================================================================
    # 4. API 路由
    # ================================================================

    @app.get("/", response_model=HealthResponse)
    async def health_check():
        """健康检查"""
        return {
            "status": "healthy",
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/api/health", response_model=HealthResponse)
    async def api_health():
        """API 健康检查"""
        return {
            "status": "healthy",
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat()
        }

    @app.post("/api/decision", response_model=DecisionResponse)
    async def single_decision(
            request: DecisionRequest,
            x_request_id: Optional[str] = Header(None),
            api: PrudenceAPI = Depends(get_prudence_api),
    ):
        """
        单次决策

        输入客户ID和产品ID，返回完整决策结果
        """
        logger = get_logger()
        start_time = time.time()
        trace_id = x_request_id or str(time.time_ns())
        try:
            logger.info(
                f"决策请求: {request.customer_id} -> {request.product_id}",
                trace_id=trace_id
            )

            result = api.decide(request.customer_id, request.product_id)
            latency_ms = (time.time() - start_time) * 1000

            # 记录决策到指标（包含延迟）
            record_decision(result, latency_ms)

            # 记录审计
            logger.audit(
                event="decision",
                customer_id=request.customer_id,
                product_id=request.product_id,
                trace_id=trace_id,
                details={"action": result["action"], "reason": result["reason"], "latency_ms": latency_ms}
            )

            logger.info(
                f"决策完成: {result['action']}",
                latency_ms=f"{latency_ms:.2f}",
                trace_id=trace_id
            )

            return DecisionResponse(**result)

        except Exception as e:
            logger.error(f"决策失败: {str(e)}", traceback=str(e), trace_id=trace_id)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/decision/batch", response_model=BatchDecisionResponse)
    async def batch_decision(
            request: BatchDecisionRequest,
            x_request_id: Optional[str] = Header(None),
            api: PrudenceAPI = Depends(get_prudence_api),
    ):
        """批量决策"""
        start_time = time.time()
        results = []
        for req in request.requests:
            try:
                result = api.decide(req.customer_id, req.product_id)
                results.append(result)
                record_decision(result)
            except Exception as e:
                results.append({
                    "customer_id": req.customer_id,
                    "product_id": req.product_id,
                    "action": "ERROR",
                    "suitability_level": "UNKNOWN",
                    "intent_score": 0.0,
                    "reason": str(e),
                    "timestamp": datetime.now().isoformat()
                })

        latency_ms = (time.time() - start_time) * 1000
        return BatchDecisionResponse(
            results=[DecisionResponse(**r) for r in results],
            total=len(results),
            latency_ms=round(latency_ms, 2)
        )

    @app.get("/api/customers")
    async def list_customers(
            api: PrudenceAPI = Depends(get_prudence_api),
    ):
        """获取所有客户列表"""
        try:
            ds = api.engines.get("data_source")
            if ds:
                customers = ds.list_customers()
                return {"customers": customers, "total": len(customers)}
            return {"customers": ["CUST_HIGH", "CUST_LOW", "CUST_ELDER"], "total": 3}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/products")
    async def list_products(
            api: PrudenceAPI = Depends(get_prudence_api),
    ):
        """获取所有产品列表"""
        try:
            ds = api.engines.get("data_source")
            if ds:
                products = ds.list_products()
                return {"products": products, "total": len(products)}
            return {"products": ["P001", "P002", "P004", "P005", "P006"], "total": 5}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/customer/{customer_id}")
    async def get_customer(
            customer_id: str,
            api: PrudenceAPI = Depends(get_prudence_api),
    ):
        """获取客户详情"""
        try:
            ds = api.engines.get("data_source")
            if ds:
                info = ds.get_customer(customer_id)
                if not info:
                    raise HTTPException(status_code=404, detail="客户不存在")
                return info
            return {"error": "未找到数据源"}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/product/{product_id}")
    async def get_product(
            product_id: str,
            api: PrudenceAPI = Depends(get_prudence_api),
    ):
        """获取产品详情"""
        try:
            ds = api.engines.get("data_source")
            if ds:
                info = ds.get_product(product_id)
                if not info:
                    raise HTTPException(status_code=404, detail="产品不存在")
                return info
            return {"error": "未找到数据源"}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/metrics", response_model=MetricsResponse)
    async def get_metrics():
        """获取决策指标"""
        from logger import get_metrics
        return get_metrics()

    @app.get("/api/audit")
    async def get_audit_logs(limit: int = 100):
        """获取审计日志"""
        from logger import get_audit_logs
        return {"logs": get_audit_logs(limit), "total": len(get_audit_logs())}

    return app


# ================================================================
# 5. 启动入口
# ================================================================

def main():
    """启动 API 服务器"""
    config = get_config()
    app = create_app(config)

    logger = get_logger()
    logger.info(
        f"启动睿衡引擎 API 服务",
        host=config.api.host,
        port=config.api.port,
        docs=f"http://{config.api.host}:{config.api.port}/api/docs"
    )

    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        log_level=config.logger.level.lower(),
        access_log=config.api.debug,
    )


if __name__ == "__main__":
    main()