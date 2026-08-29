import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

SYS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SYS_PATH)

from src.data_loader import AvocadoDataLoader
from src.price_optimizer import AvocadoPriceOptimizer

app = FastAPI(title="Prescriptive Price Optimization API", version="1.0.0")

# Initialize Data and Model
loader = AvocadoDataLoader()
df, X, y = loader.load_data(product_type="conventional", region="California")
optimizer = AvocadoPriceOptimizer(df, X, y, unit_cost=0.60)
model_results = optimizer.fit_demand_model()


class PricingRequest(BaseModel):
    region: str = Field(default="California")
    month: int = Field(default=6, ge=1, le=12)
    unit_cost: float = Field(default=0.60, ge=0.01)
    max_supply: float = Field(default=3000000.0, ge=1000.0)
    min_price: float = Field(default=0.70, ge=0.10)
    max_price: float = Field(default=3.00, le=10.0)


@app.get("/health")
def health_check():
    return {
        "status": "HEALTHY",
        "model": "Constant Elasticity Log-Log OLS Demand Model",
        "price_elasticity": model_results.get("price_elasticity_beta", -1.2258),
        "r_squared": model_results.get("r_squared", 0.753)
    }


@app.post("/prescribe_price")
def prescribe_price(req: PricingRequest):
    try:
        opt_res = optimizer.optimize_price(
            month=req.month,
            max_supply_capacity=req.max_supply,
            price_bounds=(req.min_price, req.max_price)
        )
        return {
            "region": req.region,
            "prescribed_optimal_price": opt_res["optimal_retail_price"],
            "forecasted_demand_volume": opt_res["expected_demand_units"],
            "projected_maximum_profit": opt_res["projected_weekly_profit"],
            "baseline_weekly_profit": opt_res["baseline_weekly_profit"],
            "profit_uplift_percentage": opt_res["profit_uplift_percentage"],
            "price_elasticity_beta": model_results.get("price_elasticity_beta", -1.2258)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
