"""
End-to-End Execution Pipeline for Avocado Predictive & Prescriptive Price Optimization.
Estimates log-log demand elasticity and maximizes retail profitability under supply limits.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import AvocadoDataLoader
from src.price_optimizer import AvocadoPriceOptimizer


def main():
    print("=" * 95)
    print("AVOCADO PRICE OPTIMIZATION VIA PREDICTIVE & PRESCRIPTIVE ANALYTICS")
    print("Benchmark: Hass Avocado Board (HAB) US Retail Data | Tech: Log-Log Elasticity + SLSQP Optimization")
    print("=" * 95)

    print("\n[1/3] Loading Hass Avocado Board weekly transaction history...")
    loader = AvocadoDataLoader(data_dir="data")
    df, X, y = loader.load_data(product_type='conventional', region='California')
    print(f"      Loaded {len(df):,} weekly market observations for Conventional Avocados (California).")
    print(f"      Mean Historical Price: ${df['AveragePrice'].mean():.2f} | Mean Weekly Volume: {df['Total_Volume'].mean():,.0f} units")

    print("\n[2/3] Fitting Econometric Demand Elasticity Model (Stage 1: Predictive)...")
    optimizer = AvocadoPriceOptimizer(df, X, y, unit_cost=0.60)
    model_res = optimizer.fit_demand_model()
    print(f"      Price Elasticity (Beta) : {model_res['price_elasticity_beta']:.4f} ({model_res['interpretation']})")
    print(f"      Goodness-of-Fit (R^2)   : {model_res['r_squared']:.4f}")

    print("\n[3/3] Solving Non-Linear Retail Price Optimization (Stage 2: Prescriptive)...")
    print("      -> Unit Procurement Cost  : $0.60 / avocado")
    print("      -> Weekly Supply Capacity : 3,000,000 avocados max")
    opt_res = optimizer.optimize_price(month=6, max_supply_capacity=3000000.0)

    print("\n" + "=" * 95)
    print("PRESCRIPTIVE PRICING & PROFIT OPTIMIZATION OUTPUT:")
    print("=" * 95)
    print(f"  * Optimal Recommended Price   : ${opt_res['optimal_retail_price']:.2f} per unit (vs Historical: ${opt_res['baseline_historical_price']:.2f})")
    print(f"  * Projected Weekly Demand     : {opt_res['expected_demand_units']:,.0f} units")
    print(f"  * Projected Weekly Revenue    : ${opt_res['projected_weekly_revenue']:,.2f}")
    print(f"  * Projected Weekly Profit     : ${opt_res['projected_weekly_profit']:,.2f}")
    print(f"  * Baseline Weekly Profit      : ${opt_res['baseline_weekly_profit']:,.2f}")
    print(f"  * Net Profit Uplift           : +{opt_res['profit_uplift_percentage']:.2f}% under optimal pricing")
    print("=" * 95)

    print("\n[CONCLUSION] Successfully reconciled supply constraints and consumer price sensitivity,")
    print(f"   yielding a +{opt_res['profit_uplift_percentage']:.2f}% weekly margin expansion.")
    print("=" * 95)


if __name__ == '__main__':
    main()
