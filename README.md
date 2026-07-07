Quant Research – JPMorgan Forage Simulation

Prototype quantitative research models built during JPMorgan Chase's Quantitative Research Virtual Experience Program on Forage. Each script addresses one task from the simulation, working with commodity price data and a retail loan book.

Contents

1. nat_gas_pricing.py — Natural Gas Price Estimator

Fits a trend + seasonal model (linear trend plus a sinusoidal annual cycle) to monthly Henry Hub-style natural gas price snapshots. Exposes NatGasPriceEstimator.estimate(date), which interpolates prices for any historical date and extrapolates up to one year into the future — capturing the winter-peak / summer-trough seasonality driven by heating demand and storage economics.

2. storage_contract_pricer.py — Storage Contract Pricing Model

Prices natural gas storage contracts: buy (inject) gas on one or more dates, hold it in storage, sell (withdraw) it later. price_storage_contract() generalizes to any number of injection/withdrawal events, tracks running inventory against rate and capacity constraints, and nets out all cash flows — purchase cost, sale revenue, injection/withdrawal handling fees, monthly storage rental, and transport costs — to return the contract's fair value.

3. credit_risk_pd_model.py — Probability of Default & Expected Loss

Trains and compares two classifiers (logistic regression and gradient boosted trees) on a 10,000-borrower retail loan book to predict probability of default (PD). PDModel.expected_loss() combines the predicted PD with a fixed loss-given-default rate (90%, i.e. 10% recovery) and the loan's exposure at default to estimate expected loss per loan.

4. fico_bucketing.py — FICO Score Bucketing via Dynamic Programming

Converts continuous FICO scores into a small number of categorical rating buckets for use as model input labels. Implements two DP-optimal quantization objectives: minimizing mean squared error (treating each bucket as its mean) and maximizing log-likelihood of observed defaults under a piecewise-constant PD model per bucket. RatingMap.rate(fico_score) returns the final integer rating (1 = best credit quality).

Running the scripts

Each script needs the accompanying loan/price data CSV in the same folder. With Python 3 and pandas, numpy, scikit-learn, and matplotlib installed:

bashpython nat_gas_pricing.py
python storage_contract_pricer.py
python credit_risk_pd_model.py
python fico_bucketing.py

Data

Sample data was provided as part of the Forage simulation: monthly natural gas prices (Oct 2020 – Sep 2024) and a synthetic retail loan book (10,000 borrowers) with FICO scores, outstanding debt, income, and default flags.
