# Prescriptive Retail Pricing & Econometric Demand Optimization Engine
> **Two-Stage Prescriptive Optimization Pipeline: Econometric Log-Log Demand Elasticity Estimation (OLS) and Non-Linear Profit Maximization under Capacity Constraints (SciPy SLSQP)**  
> *Operations Research · Econometric Elasticity · Non-Linear Constrained Optimization · SciPy SLSQP · FastAPI Microservices*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/SurajChouhan14/Avocado-Price-and-Demand-Prescriptive-Optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/SurajChouhan14/Avocado-Price-and-Demand-Prescriptive-Optimizer/actions)
[![Benchmark](https://img.shields.io/badge/benchmark-HAB%20Retail%20(470%20weeks)-blue.svg)]()
[![Tests](https://img.shields.io/badge/tests-5%20passed-brightgreen.svg)]()

---

## 🎯 Executive Overview & Architecture
Pricing decision-making in multi-echelon retail requires balancing **consumer price elasticity**, **seasonal demand fluctuations**, and **upstream supply capacity limits**. Setting prices without optimization risks margin erosion (underpricing) or unsold surplus inventory (overpricing).

This repository implements a **Two-Stage Predictive & Prescriptive Retail Optimization Engine**:
1. **Stage 1 (Predictive Econometrics):** Fits a constant-elasticity log-log demand model with annual Fourier seasonality over 9 years of weekly Hass Avocado Board retail transactions:
   $$\ln Q(P) = lpha + eta \ln P + \gamma_1 \sin\left(rac{2\pi M}{12}ight) + \gamma_2 \cos\left(rac{2\pi M}{12}ight)$$
2. **Stage 2 (Prescriptive Non-Linear Optimization):** Solves the constrained profit maximization program via **SciPy SLSQP (Sequential Least Squares Programming)**:
   $$\max_{P} \; (P - c) \cdot Q(P) \quad 	ext{s.t.} \quad Q(P) \le 	ext{MaxSupply}, \quad P_{\min} \le P \le P_{\max}$$

```
  ┌────────────────────────────────────────────────────────┐
  │ Hass Avocado Board (HAB) Retail Dataset (470 Weeks)    │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ Stage 1: Econometric OLS Demand Elasticity Fit         │
  │ • Beta = -1.2258 (Elastic Demand, |Beta| > 1)          │
  │ • R^2 = 0.7530 (Annual Fourier Seasonality)            │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ Stage 2: SciPy SLSQP Non-Linear Profit Optimizer       │
  │ • Analytic Monopoly Optimum: P* = (Beta/(1+Beta))*c    │
  │ • Unconstrained Interior Solution: P* = $3.26 (+34.9%) │
  │ • Operational Retail Price Ceiling: P* = $2.50 (+33.5%)│
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ Production FastAPI Microservice Endpoint               │
  │ • Real-time Prescriptive Pricing at Sub-10ms Latency   │
  └────────────────────────────────────────────────────────┘
```

---

## 📊 Benchmark Execution & Validation Report

### Hass Avocado Board (HAB) California Conventional Benchmark (470 Weeks, 2015–2023)

| Metric | Measured Value | Verification Method | Operational Definition |
|---|:---:|:---:|---|
| **Data Scale & Lineage** | **$470	ext{ Weekly Observations}$** | SHA-256 (`631e7701...`) | Official Hass Avocado Board retail dataset, California conventional slice (2015–2023) |
| **Historical Baseline Price** | **$\$1.1490	ext{ per unit}$** | Historical Dataset Mean | Mean observed retail price across all historical weeks ($pprox \$1.15$) |
| **Price Elasticity ($eta$)** | **$-1.2258$** | OLS Normal Equations | Elastic consumer demand ($|eta| > 1.0$); 1% price increase yields 1.23% demand decline |
| **Model Goodness-of-Fit ($R^2$)** | **$0.7530$** | Regression Coefficient | Explains 75.3% of weekly demand variance across price and seasonal cycles |
| **Unconstrained Optimum ($P^*$)** | **$\$3.2562	ext{ (~}\$3.26	ext{)}$** | SciPy SLSQP vs. Analytic Theory | Matches closed-form theoretical monopoly price $P^* = rac{eta}{1+eta}c = \$3.2567$ within $0.015\%$ |
| **Unconstrained Profit Uplift** | **$+34.94\%$** | In-Sample Model Projection | Model-projected weekly profit expansion over historical baseline ($P^* = \$3.26$, slack capacity) |
| **Operational Price Cap ($P^*$)** | **$\$2.5000	ext{ (Bound-Active)}$** | Bound-Constrained SLSQP | Under operational retail price cap ($P \le \$2.50$), solver lands on boundary with $+33.45\%$ uplift |

*Note: All profit uplift figures are in-sample, model-projected under estimated constant-elasticity demand with no holdout evaluation.*

---

## 📁 Repository Structure

```text
Avocado-Price-and-Demand-Prescriptive-Optimizer/
├── .github/
│   └── workflows/
│       └── ci.yml                  # Automated test & benchmark CI workflow
├── data/
│   └── avocado.csv                 # Hass Avocado Board retail transaction dataset
├── results/
│   └── final_benchmark.txt         # Frozen execution output & reproducibility log
├── src/
│   ├── data_loader.py              # Data ingestion & Fourier feature preparation
│   └── price_optimizer.py          # OLS demand fitting & SLSQP profit optimizer
├── requirements.txt                # Dependencies (scipy, numpy, pandas, fastapi, uvicorn)
├── run_pipeline.py                 # Full benchmark runner & results logger
├── serve_api.py                    # Production FastAPI REST microservice
└── test_avocado_pricing.py         # 5 hard unit & mathematical validation tests
```

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/SurajChouhan14/Avocado-Price-and-Demand-Prescriptive-Optimizer.git
cd Avocado-Price-and-Demand-Prescriptive-Optimizer
pip install -r requirements.txt
```

### 2. Run Pipeline Benchmark
```bash
python run_pipeline.py
```

### 3. Run Test Suite
```bash
python test_avocado_pricing.py
```

### 4. Serve API Microservice
```bash
uvicorn serve_api:app --host 0.0.0.0 --port 8000
```
