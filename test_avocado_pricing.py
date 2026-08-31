"""
Automated Test Suite for Avocado Predictive & Prescriptive Price Optimization Engine.
Tests:
1. Dataset Ingestion & SHA-256 Checksum Validation
2. Econometric OLS Demand Elasticity & Monotonicity
3. Unconstrained SLSQP match with Analytic Monopoly Optimum (0.5% tolerance)
4. Supply Capacity Constraint Binding Detection
5. Operational Price Ceiling Bound-Active Flag Detection
"""

import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import AvocadoDataLoader
from src.price_optimizer import AvocadoPriceOptimizer


class TestAvocadoPriceOptimizer(unittest.TestCase):
    """
    Hard unit tests for econometric demand modeling and non-linear SLSQP optimization.
    """

    @classmethod
    def setUpClass(cls):
        cls.loader = AvocadoDataLoader(data_dir="data")
        cls.df, cls.X, cls.y = cls.loader.load_data(product_type='conventional', region='California')
        cls.optimizer = AvocadoPriceOptimizer(cls.df, cls.X, cls.y, unit_cost=0.60)
        cls.model_res = cls.optimizer.fit_demand_model()

    def test_1_dataset_ingestion_and_checksum(self):
        """Verify HAB dataset presence, row counts (470 California weeks), and SHA-256 fingerprint."""
        self.assertTrue(self.loader.validate_checksum(), "SHA-256 checksum mismatch on raw avocado.csv!")
        self.assertEqual(len(self.df), 470, f"Expected 470 California conventional records, got {len(self.df)}")
        self.assertEqual(self.X.shape, (470, 3))
        self.assertEqual(len(self.y), 470)

    def test_2_econometric_elasticity_and_monotonicity(self):
        """Verify negative price elasticity beta < -1.0 (elastic demand) and R2 > 0.70."""
        beta = self.model_res['price_elasticity_beta']
        r2 = self.model_res['r_squared']
        self.assertLess(beta, -1.0, f"Expected elastic demand beta < -1.0, got {beta}")
        self.assertGreater(r2, 0.70, f"Expected R2 > 0.70, got {r2}")

        # Test demand monotonicity (higher price -> strictly lower demand)
        q1 = self.optimizer.predict_demand(price=1.00, month=6)
        q2 = self.optimizer.predict_demand(price=2.00, month=6)
        q3 = self.optimizer.predict_demand(price=3.00, month=6)
        self.assertGreater(q1, q2)
        self.assertGreater(q2, q3)

    def test_3_unconstrained_slsqp_matches_analytic_optimum(self):
        """Verify wide-bounds SLSQP finds the interior economic optimum matching closed-form formula within 0.5%."""
        p_analytic = self.optimizer.get_analytic_optimum_price(unit_cost=0.60)
        self.assertAlmostEqual(p_analytic, 3.2567, places=3)

        # Run unconstrained SLSQP
        res = self.optimizer.optimize_price(month=6, max_supply_capacity=5000000.0, price_bounds=(0.50, 5.00))
        p_slsqp = res['optimal_retail_price']

        rel_error = abs(p_slsqp - p_analytic) / p_analytic
        self.assertLess(rel_error, 0.005, f"SLSQP price ${p_slsqp:.4f} differs from analytic ${p_analytic:.4f} by {rel_error*100:.3f}%")
        self.assertFalse(res['price_bound_active'], "Price bound should NOT be active for wide bounds")
        self.assertEqual(res['solution_type'], "INTERIOR_OPTIMUM")
        self.assertGreater(res['profit_uplift_percentage'], 30.0)

    def test_4_supply_capacity_constraint_binding(self):
        """Verify that tight supply capacity binds and forces price higher to clear market."""
        res_tight = self.optimizer.optimize_price(month=6, max_supply_capacity=500000.0, price_bounds=(0.50, 6.00))
        self.assertTrue(res_tight['capacity_constraint_active'], "Capacity constraint should be active when supply is 500k")
        self.assertEqual(res_tight['solution_type'], "CAPACITY_CONSTRAINED")
        self.assertAlmostEqual(res_tight['expected_demand_units'], 500000.0, delta=1000.0)
        self.assertGreater(res_tight['optimal_retail_price'], 3.26)

    def test_5_operational_price_bound_binding(self):
        """Verify that operational retail price cap ($2.50) binds and sets price_bound_active=True."""
        res_cap = self.optimizer.optimize_price(month=6, max_supply_capacity=3000000.0, price_bounds=(0.70, 2.50))
        self.assertTrue(res_cap['price_bound_active'], "Price bound should be ACTIVE when cap is $2.50")
        self.assertEqual(res_cap['solution_type'], "BOUND_CONSTRAINED")
        self.assertEqual(res_cap['optimal_retail_price'], 2.5000)
        self.assertAlmostEqual(res_cap['profit_uplift_percentage'], 33.45, delta=0.5)


if __name__ == '__main__':
    unittest.main()
