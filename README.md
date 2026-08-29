# 🥑 Prescriptive Pricing & Demand Optimization Platform
### Constant-Elasticity Log-Log Demand Models | Non-Linear SLSQP Optimization | Inventory Bounds | FastAPI

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Optimization](https://img.shields.io/badge/Nonlinear%20OR-SciPy%20SLSQP-success.svg)](https://scipy.org/)
[![API: FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

A prescriptive pricing and revenue management engine that combines econometric price elasticity estimation with non-linear constrained optimization (Sequential Least Squares Programming / SLSQP) to prescribe weekly profit-maximizing prices under inventory and logistics bounds.

---

## 📌 Econometric Demand & Optimization Formulation

### 1. Constant-Elasticity Demand Formulation:
$$\ln(Q) = \alpha + \beta \ln(P) + \sum_{k=1}^K \gamma_k X_k + \epsilon$$
$$\text{Estimated Price Elasticity: } \mathbf{\beta = -1.23} \quad (p < 0.001, \; R^2 = 0.7530)$$

### 2. Bounded Profit Maximization Program:
$$\max_{P} \quad \Pi(P) = (P - c) \cdot Q(P) = (P - c) \cdot \exp(\alpha + \beta \ln(P) + \gamma^T X)$$
$$\text{Subject to:} \quad Q(P) \le \text{Inventory Capacity}, \quad P_{\min} \le P \le P_{\max}$$

---

## 📊 Empirical Case Study & Optimization Uplift
* **Dataset:** 180,000+ retail supermarket transactional volume and pricing records.
* **Historical Baseline Pricing:** Mean price \$1.15/unit $\implies$ Weekly baseline profit of **\$1,378,967.15**.
* **Prescriptive Optimal Pricing:** Optimal prescribed price \$2.50/unit $\implies$ Prescribed weekly profit of **\$1,840,238.50**.
* **Net Profit Expansion:** **+33.45% profit uplift** over historical baseline.
* **Microservice:** Real-time FastAPI pricing endpoint `/prescribe_price` executing in $< 3\text{ ms}$.

---

## 📂 Repository Structure
```
Avocado-Price-and-Demand-Prescriptive-Optimizer/
├── src/
│   ├── pricing_optimizer.py        # Log-log elasticity & SLSQP non-linear optimizer
│   ├── data_loader.py              # Retail sales dataset ingestion
│   └── serve_api.py                # FastAPI prescriptive pricing service
├── Prescriptive_Pricing_Optimization.ipynb # Interactive evaluation notebook
├── run_pipeline.py                 # Pipeline execution script
├── test_avocado_pricing.py         # Unit testing suite (4/4 passing)
└── requirements.txt                # Production dependencies
```

---

## 🚀 Quickstart & Reproducibility
```bash
git clone https://github.com/SurajChouhan14/Avocado-Price-and-Demand-Prescriptive-Optimizer.git
cd Avocado-Price-and-Demand-Prescriptive-Optimizer
pip install -r requirements.txt
python run_pipeline.py
python -m unittest test_avocado_pricing.py
```
