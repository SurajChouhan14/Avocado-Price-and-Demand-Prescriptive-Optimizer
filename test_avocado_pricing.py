"""
Automated Unit Test Suite for Avocado Predictive & Prescriptive Price Optimization Engine.
Verifies Data Loading, Elasticity Estimation, Demand Monotonicity, and Profit Uplift.
"""

import unittest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import AvocadoDataLoader
from src.price_optimizer import AvocadoPriceOptimizer


class TestAvocadoPriceOptimizer(unittest.TestCase):
    """
    Unit test cases for avocado price elasticity and prescriptive optimization.
    """

    @classmethod
    def setUpClass(cls):
        cls.loader = AvocadoDataLoader(data_dir="data")
        cls.df, cls.X, cls.y = cls.loader.load_data(product_type='conventional', region='California')
        cls.optimizer = AvocadoPriceOptimizer(cls.df, cls.X, cls.y, unit_cost=0.60)
        cls.model_res = cls.optimizer.fit_demand_model()
        cls.opt_res = cls.optimizer.optimize_price(month=6, max_supply_capacity=3000000.0)

    def test_data_loading_and_features(self):
        """Verify avocado dataset loading and design matrix shape."""
        self.assertGreater(len(self.df), 100)
        self.assertEqual(self.X.shape[1], 3)
        self.assertEqual(len(self.y), len(self.df))

    def test_elasticity_sign_and_magnitude(self):
        """Verify negative price elasticity (law of demand)."""
        self.assertLess(self.model_res['price_elasticity_beta'], 0.0)
        self.assertGreater(self.model_res['r_squared'], 0.50)

    def test_demand_monotonicity(self):
        """Verify higher prices result in strictly lower predicted demand."""
        q_low_p = self.optimizer.predict_demand(price=1.00, month=6)
        q_high_p = self.optimizer.predict_demand(price=2.00, month=6)
        self.assertGreater(q_low_p, q_high_p)

    def test_prescriptive_profit_uplift(self):
        """Verify optimal pricing yields positive profit and uplift over baseline."""
        self.assertGreater(self.opt_res['projected_weekly_profit'], 0.0)
        self.assertGreater(self.opt_res['profit_uplift_percentage'], 0.0)
        self.assertGreaterEqual(self.opt_res['optimal_retail_price'], 0.60)


if __name__ == '__main__':
    unittest.main()
