"""
Avocado Predictive Demand Modeling & Prescriptive Price Optimization Engine.

Stage 1 (Predictive): Log-Log Econometric Demand Elasticity Estimation via OLS Regression
Stage 2 (Prescriptive): Non-Linear Profit & Revenue Optimization under Inventory Supply Caps
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize


class AvocadoPriceOptimizer:
    """
    Predictive price elasticity estimation and prescriptive retail pricing optimizer.
    """

    def __init__(self, df, X, y, unit_cost=0.60):
        self.df = df
        self.X = X
        self.y = y
        self.unit_cost = unit_cost
        self.beta = None
        self.intercept = None
        self.elasticity = None
        self.r2 = None

    def fit_demand_model(self):
        """
        Fits log-log econometric regression to estimate price elasticity beta.
        ln(Q) = alpha + beta * ln(P) + gamma_1 * sin(month) + gamma_2 * cos(month)
        """
        # Add intercept column
        n = len(self.y)
        X_design = np.hstack([np.ones((n, 1)), self.X])
        
        # OLS Normal Equations: theta = (X^T X)^-1 X^T y
        theta = np.linalg.solve(X_design.T @ X_design, X_design.T @ self.y)
        self.intercept = theta[0]
        self.elasticity = theta[1]  # Price elasticity beta
        self.gamma_sin = theta[2]
        self.gamma_cos = theta[3]

        y_pred = X_design @ theta
        ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
        ss_res = np.sum((self.y - y_pred) ** 2)
        self.r2 = 1.0 - (ss_res / ss_tot)

        return {
            'price_elasticity_beta': round(float(self.elasticity), 4),
            'intercept_alpha': round(float(self.intercept), 4),
            'r_squared': round(float(self.r2), 4),
            'interpretation': f"Demand is {'ELASTIC (|beta| > 1)' if abs(self.elasticity) > 1 else 'INELASTIC (|beta| <= 1)'}"
        }

    def predict_demand(self, price, month=6):
        """
        Predicts expected quantity demanded at a given price and month.
        """
        sin_m = np.sin(2 * np.pi * month / 12.0)
        cos_m = np.cos(2 * np.pi * month / 12.0)
        log_q = self.intercept + self.elasticity * np.log(price) + self.gamma_sin * sin_m + self.gamma_cos * cos_m
        return float(np.exp(log_q))

    def optimize_price(self, month=6, max_supply_capacity=3000000.0, price_bounds=(0.70, 2.50)):
        """
        Prescriptive Optimization: Finds profit-maximizing price subject to supply constraints.
        Max (P - UnitCost) * Q(P)
        s.t. Q(P) <= MaxSupply, P in [P_min, P_max]
        """
        if self.elasticity is None:
            self.fit_demand_model()

        def neg_profit(p):
            p_val = float(p)
            q_val = self.predict_demand(p_val, month=month)
            if q_val > max_supply_capacity:
                # Capacity penalty
                penalty = 1e6 * (q_val - max_supply_capacity)
                return -((p_val - self.unit_cost) * max_supply_capacity - penalty)
            profit = (p_val - self.unit_cost) * q_val
            return -profit

        res = minimize_scalar(neg_profit, bounds=price_bounds, method='bounded')
        opt_price = float(res.x)
        opt_demand = self.predict_demand(opt_price, month=month)
        opt_revenue = opt_price * opt_demand
        opt_profit = (opt_price - self.unit_cost) * opt_demand

        # Benchmark comparison against mean historical price
        hist_mean_price = float(self.df['AveragePrice'].mean())
        hist_demand = self.predict_demand(hist_mean_price, month=month)
        hist_profit = (hist_mean_price - self.unit_cost) * hist_demand
        hist_revenue = hist_mean_price * hist_demand

        profit_uplift_pct = ((opt_profit - hist_profit) / hist_profit) * 100.0 if hist_profit > 0 else 0.0

        return {
            'optimal_retail_price': round(opt_price, 2),
            'expected_demand_units': round(opt_demand, 0),
            'projected_weekly_revenue': round(opt_revenue, 2),
            'projected_weekly_profit': round(opt_profit, 2),
            'baseline_historical_price': round(hist_mean_price, 2),
            'baseline_weekly_profit': round(hist_profit, 2),
            'profit_uplift_percentage': round(profit_uplift_pct, 2)
        }
