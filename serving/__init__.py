"""TrustMesh serving layer: loads the already-trained Graph and Voice models and serves them.

Nothing in this package trains a model. Scores come from the real XGBoost Tier-1+2 weights
(`artifacts/graph/xgboost_tier1_tier2.json`) and the real frozen-SSL voice heads
(`artifacts/voice/*.pt`); `fusion.py` combines the two independent scores.
"""
