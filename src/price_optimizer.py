"""
Avocado Econometric Demand Modeling & Prescriptive Pricing Optimizer.

Stage 1 (Predictive): Log-Log Demand Elasticity Estimation via OLS Normal Equations
Stage 2 (Prescriptive): Non-Linear Profit Maximization under Capacity Limits & Price Bounds (SciPy SLSQP)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class AvocadoPriceOptimizer:
    """
    Predictive price elasticity estimation and prescriptive retail pricing optimizer
    powered by Econometric OLS and SciPy SLSQP (Sequential Least Squares Programming).
    """

    def __init__(self, df, X, y, unit_cost=0.60):
        self.df = df
        self.X = X
        self.y = y
        self.unit_cost = float(unit_cost)
        self.intercept = None
        self.elasticity = None
        self.gamma_sin = None
        self.gamma_cos = None
        self.r2 = None

    def fit_demand_model(self):
        """
        Fits log-log econometric regression to estimate price elasticity beta.
        ln(Q) = alpha + beta * ln(P) + gamma_1 * sin(2*pi*month/12) + gamma_2 * cos(2*pi*month/12)
        """
        n = len(self.y)
        X_design = np.hstack([np.ones((n, 1)), self.X])

        # OLS Normal Equations: theta = (X^T X)^-1 X^T y
        theta = np.linalg.solve(X_design.T @ X_design, X_design.T @ self.y)
        self.intercept = float(theta[0])
        self.elasticity = float(theta[1])  # Price elasticity beta
        self.gamma_sin = float(theta[2])
        self.gamma_cos = float(theta[3])

        y_pred = X_design @ theta
        ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
        ss_res = np.sum((self.y - y_pred) ** 2)
        self.r2 = float(1.0 - (ss_res / ss_tot))

        return {
            'price_elasticity_beta': round(self.elasticity, 4),
            'intercept_alpha': round(self.intercept, 4),
            'seasonality_sin': round(self.gamma_sin, 4),
            'seasonality_cos': round(self.gamma_cos, 4),
            'r_squared': round(self.r2, 4),
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

    def get_analytic_optimum_price(self, unit_cost=None):
        """
        Computes closed-form theoretical monopoly price: P* = (beta / (1 + beta)) * c.
        """
        if self.elasticity is None:
            self.fit_demand_model()
        c = float(unit_cost) if unit_cost is not None else self.unit_cost
        if self.elasticity >= -1.0:
            raise ValueError("Analytic optimum requires elastic demand (beta < -1.0)")
        return float((self.elasticity / (1.0 + self.elasticity)) * c)

    def optimize_price(self, month=6, max_supply_capacity=5000000.0, price_bounds=(0.50, 5.00), unit_cost=None):
        """
        Prescriptive Optimization using SciPy SLSQP.
        Max (P - UnitCost) * Q(P)
        s.t. Q(P) <= MaxSupply, P in [P_min, P_max]
        """
        if self.elasticity is None:
            self.fit_demand_model()

        c = float(unit_cost) if unit_cost is not None else self.unit_cost
        sin_m = np.sin(2 * np.pi * month / 12.0)
        cos_m = np.cos(2 * np.pi * month / 12.0)

        # Scaled Objective Function (in Millions USD) for stable SLSQP convergence
        def scaled_neg_profit(p_vec):
            price = p_vec[0]
            log_q = self.intercept + self.elasticity * np.log(price) + self.gamma_sin * sin_m + self.gamma_cos * cos_m
            q = np.exp(log_q)
            profit = (price - c) * q
            return -(profit / 1e6)

        # Non-linear Inequality Constraint: max_supply_capacity - Q(P) >= 0 (scaled)
        constraints = [{
            'type': 'ineq',
            'fun': lambda p_vec: (max_supply_capacity - np.exp(
                self.intercept + self.elasticity * np.log(p_vec[0]) + self.gamma_sin * sin_m + self.gamma_cos * cos_m
            )) / 1e6
        }]

        hist_mean_price = float(self.df['AveragePrice'].mean())
        p0 = [hist_mean_price]
        bounds = [price_bounds]

        # Execute SciPy SLSQP Solver
        res = minimize(scaled_neg_profit, p0, method='SLSQP', bounds=bounds, constraints=constraints)

        opt_price = float(res.x[0])
        opt_demand = self.predict_demand(opt_price, month=month)
        opt_revenue = opt_price * opt_demand
        opt_profit = (opt_price - c) * opt_demand

        # Baseline benchmark against historical mean price ($1.1490)
        hist_demand = self.predict_demand(hist_mean_price, month=month)
        hist_profit = (hist_mean_price - c) * hist_demand
        hist_revenue = hist_mean_price * hist_demand

        profit_uplift_pct = ((opt_profit - hist_profit) / hist_profit) * 100.0 if hist_profit > 0 else 0.0

        # Constraint & Bound Activity Diagnostics
        is_price_bound_active = bool(
            abs(opt_price - price_bounds[0]) < 1e-3 or abs(opt_price - price_bounds[1]) < 1e-3
        )
        is_capacity_active = bool(
            abs(opt_demand - max_supply_capacity) / max_supply_capacity < 1e-3
        )
        analytic_opt = self.get_analytic_optimum_price(unit_cost=c)

        solution_type = "BOUND_CONSTRAINED" if is_price_bound_active else (
            "CAPACITY_CONSTRAINED" if is_capacity_active else "INTERIOR_OPTIMUM"
        )

        return {
            'optimization_solver': 'SciPy SLSQP',
            'solver_status': res.message,
            'slsqp_iterations': int(res.nit),
            'solution_type': solution_type,
            'optimal_retail_price': round(opt_price, 4),
            'theoretical_analytic_price': round(analytic_opt, 4),
            'price_bound_active': is_price_bound_active,
            'capacity_constraint_active': is_capacity_active,
            'expected_demand_units': round(opt_demand, 2),
            'projected_weekly_revenue': round(opt_revenue, 2),
            'projected_weekly_profit': round(opt_profit, 2),
            'baseline_historical_price': round(hist_mean_price, 4),
            'baseline_weekly_profit': round(hist_profit, 2),
            'profit_uplift_percentage': round(profit_uplift_pct, 2)
        }
