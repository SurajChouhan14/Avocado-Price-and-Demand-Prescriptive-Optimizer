# Avocado Price & Demand Prescriptive Optimization Engine

An end-to-end Machine Learning and Operations Research system combining **Predictive Econometric Demand Elasticity Modeling** and **Prescriptive Non-Linear Profit Optimization** on the canonical **Hass Avocado Board (HAB) US Retail Benchmark**.

---

## 1. System Architecture

```
                                 +-------------------------------------+
                                 | Hass Avocado Board Retail Dataset   |
                                 | (Weekly Volume, Prices, Bag Sizes)  |
                                 +------------------+------------------+
                                                    |
                                                    v
                                 +-------------------------------------+
                                 | Stage 1: Predictive Demand Model    |
                                 | ln(Q) = alpha + beta*ln(P) + Season |
                                 | (Price Elasticity Estimation)       |
                                 +------------------+------------------+
                                                    | Demand Curve Q(P)
                                                    v
                                 +-------------------------------------+
                                 | Stage 2: Prescriptive Optimizer     |
                                 | Max (P - Cost) * Q(P)               |
                                 | s.t. Q(P) <= Max Weekly Supply Cap  |
                                 +------------------+------------------+
                                                    |
                                                    v
                                 +-------------------------------------+
                                 | Optimal Retail Pricing Schedule &   |
                                 | Net Margin Uplift Strategy          |
                                 +-------------------------------------+
```

---

## 2. Mathematical Formulation

### **Stage 1 (Predictive Econometric Demand Curve)**:
$$\ln(Q_{t}) = \alpha + \beta \ln(P_{t}) + \gamma_1 \sin\left(\frac{2\pi m_t}{12}\right) + \gamma_2 \cos\left(\frac{2\pi m_t}{12}\right) + \epsilon_t$$
* Estimated Price Elasticity: $\beta = -1.23$ ($p < 0.001$), indicating elastic consumer demand.

### **Stage 2 (Prescriptive Non-Linear Optimization)**:
$$\max_{P} \quad \Pi(P) = (P - C) \cdot \hat{Q}(P)$$
$$\text{subject to } \quad \hat{Q}(P) \le \text{SupplyCap}, \quad P \in [P_{\min}, P_{\max}]$$
* Where $C = \$0.60/\text{unit}$ is the unit procurement and handling cost.

---

## 3. Exact Computed Benchmark Results (California Market)

```
===============================================================================================
AVOCADO PREDICTIVE & PRESCRIPTIVE PRICING OUTPUT
===============================================================================================
  * Estimated Price Elasticity  : -1.2258 (Elastic Consumer Demand)
  * Goodness-of-Fit (R^2)       : 0.7530
  * Baseline Historical Price   : $1.15 per avocado (Baseline Profit: $1,378,967.15/wk)
  * Prescriptive Optimal Price  : $2.50 per avocado
  * Projected Weekly Profit     : $1,840,238.15 per week
  * Net Margin Uplift           : +33.45% profit expansion under optimal pricing
===============================================================================================
```

---

## 4. Quick Start & Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run price optimization pipeline
python run_pipeline.py

# 3. Run automated unit tests
python test_avocado_pricing.py
```

---

## 5. Master Placement Resume Description

> **Avocado Price Optimization (Predictive & Prescriptive Analytics)**
> * Developed an end-to-end pricing analytics engine on 6,500+ Hass Avocado Board transactions combining log-log demand elasticity modeling with non-linear prescriptive profit optimization.
> * Estimated empirical price elasticity of demand ($\beta = -1.23, R^2 = 0.75$) with harmonic seasonality decomposition.
> * Formulated a constrained revenue optimization program under inventory supply limits, unlocking a **+33.5% weekly margin expansion**.

---

## License
MIT License. Open for academic research and portfolio demonstration.
