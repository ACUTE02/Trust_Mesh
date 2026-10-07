"""XGBoost baselines, refit from the best hyperparameters found by the Phase 1 / 1B Optuna searches."""
import xgboost as xgb

# Best params from the Optuna searches in notebooks/01_graph_baseline.ipynb (recorded in
# artifacts/graph/results.json). scale_pos_weight is recomputed from the training split, not stored.
XGB_TIER1_PARAMS = {
    "max_depth": 7,
    "eta": 0.11098926863362221,
    "subsample": 0.8454933633696089,
    "colsample_bytree": 0.7300523345284323,
    "min_child_weight": 5,
}
XGB_TIER12_PARAMS = {
    "max_depth": 4,
    "eta": 0.0858855758022927,
    "subsample": 0.7582529708241003,
    "colsample_bytree": 0.9860930071516779,
    "min_child_weight": 3,
}


def fit_xgb(X_train, y_train, params, pos_weight, seed):
    model = xgb.XGBClassifier(
        **params,
        scale_pos_weight=pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        seed=seed,
        n_estimators=300,
    )
    model.fit(X_train, y_train)
    return model
