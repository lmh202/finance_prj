"""Risk Engine — HAR volatility forecast + Filtered Historical Simulation → VaR/ES.

Online inference over the model trained offline (data/processed/risk_model.json).
Public surface: engine.risk_estimates, engine.portfolio_risk, engine.model_available,
and the RiskEstimate / PortfolioRisk dataclasses.
"""
