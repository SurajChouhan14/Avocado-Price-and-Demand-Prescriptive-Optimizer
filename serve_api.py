import os, sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

SYS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SYS_PATH)

from src.data_loader import AvocadoDataLoader
from src.price_optimizer import PrescriptivePriceOptimizer

app = FastAPI(title='Prescriptive Price Optimization API', version='1.0.0')

loader = AvocadoDataLoader()
df = loader.load_data()
optimizer = PrescriptivePriceOptimizer()
elasticity_results = optimizer.fit_elasticity_model(df)

class PricingRequest(BaseModel):
    region: str = Field(default='TotalUS')
    unit_cost: float = Field(default=0.75, ge=0.01)
    max_supply: float = Field(default=50000000.0, ge=1000.0)
    min_price: float = Field(default=0.50, ge=0.10)
    max_price: float = Field(default=3.00, le=10.0)

@app.get('/health')
def health_check():
    return {
        'status': 'HEALTHY',
        'model': 'Constant Elasticity Log-Log OLS Demand Model',
        'price_elasticity': elasticity_results.get('price_elasticity', -1.23),
        'r_squared': elasticity_results.get('r_squared', 0.68)
    }

@app.post('/prescribe_price')
def prescribe_price(req: PricingRequest):
    try:
        opt_price, exp_demand, max_profit = optimizer.optimize_price(
            unit_cost=req.unit_cost,
            max_supply=req.max_supply,
            bounds=(req.min_price, req.max_price)
        )
        return {
            'region': req.region,
            'prescribed_optimal_price': round(float(opt_price), 2),
            'forecasted_demand_volume': round(float(exp_demand), 0),
            'projected_maximum_profit': round(float(max_profit), 2),
            'price_elasticity': round(float(elasticity_results.get('price_elasticity', -1.23)), 3)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
