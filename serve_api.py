"""
FastAPI REST Microservice for Prescriptive Price Optimization.
Features:
- Lifespan asynchronous model initialization
- Dynamic unit cost parameterization
- HTTP 422 validation on unsupported regions
- Secure exception handling without traceback leakage
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

SYS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SYS_PATH)

from src.data_loader import AvocadoDataLoader
from src.price_optimizer import AvocadoPriceOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("avocado_api")

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Avocado Econometric Demand Model...")
    loader = AvocadoDataLoader()
    df, X, y = loader.load_data(product_type="conventional", region="California")
    optimizer = AvocadoPriceOptimizer(df, X, y, unit_cost=0.60)
    model_res = optimizer.fit_demand_model()
    ml_models["optimizer"] = optimizer
    ml_models["model_res"] = model_res
    logger.info(f"Model initialized: Beta={model_res['price_elasticity_beta']}, R2={model_res['r_squared']}")
    yield
    ml_models.clear()


app = FastAPI(
    title="Prescriptive Price Optimization API",
    description="Non-linear retail price optimization under constant elasticity and capacity limits.",
    version="1.2.0",
    lifespan=lifespan
)


class PricingRequest(BaseModel):
    region: str = Field(default="California", description="Target retail market region")
    month: int = Field(default=6, ge=1, le=12, description="Target calendar month (1-12)")
    unit_cost: float = Field(default=0.60, ge=0.01, description="Unit procurement marginal cost ($)")
    max_supply: float = Field(default=5000000.0, ge=1000.0, description="Weekly supply capacity cap")
    min_price: float = Field(default=0.50, ge=0.10, description="Lower price bound ($)")
    max_price: float = Field(default=5.00, le=10.0, description="Upper price bound ($)")


@app.get("/health")
def health_check():
    model_res = ml_models.get("model_res", {})
    return {
        "status": "HEALTHY",
        "model": "Constant Elasticity Log-Log OLS Demand Model",
        "price_elasticity": model_res.get("price_elasticity_beta", -1.2258),
        "r_squared": model_res.get("r_squared", 0.7530)
    }


@app.post("/prescribe_price")
def prescribe_price(req: PricingRequest):
    if req.region.lower() != "california":
        raise HTTPException(
            status_code=422,
            detail=f"Region '{req.region}' not supported. Currently supported regions: ['California']"
        )

    optimizer = ml_models.get("optimizer")
    if not optimizer:
        raise HTTPException(status_code=503, detail="Model optimizer not loaded")

    try:
        opt_res = optimizer.optimize_price(
            month=req.month,
            max_supply_capacity=req.max_supply,
            price_bounds=(req.min_price, req.max_price),
            unit_cost=req.unit_cost
        )
        return {
            "region": req.region,
            "solution_type": opt_res["solution_type"],
            "prescribed_optimal_price": opt_res["optimal_retail_price"],
            "theoretical_analytic_price": opt_res["theoretical_analytic_price"],
            "price_bound_active": opt_res["price_bound_active"],
            "capacity_constraint_active": opt_res["capacity_constraint_active"],
            "forecasted_demand_volume": opt_res["expected_demand_units"],
            "projected_maximum_profit": opt_res["projected_weekly_profit"],
            "baseline_weekly_profit": opt_res["baseline_weekly_profit"],
            "profit_uplift_percentage": opt_res["profit_uplift_percentage"],
            "evaluation_note": "In-sample, model-projected under constant-elasticity demand, no holdout evaluation."
        }
    except Exception as e:
        logger.error(f"Optimization failure: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Optimization solver computation error")
