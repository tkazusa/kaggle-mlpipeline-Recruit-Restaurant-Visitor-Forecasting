import argparse
import json

import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    # Path to dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    args, _ = parser.parse_known_args()
    X_train = pd.read_csv(args.input_dir + "/X_train.csv")
    y_train = pd.read_csv(args.input_dir + "/y_train.csv")
    X_test = pd.read_csv(args.input_dir + "/X_test.csv")
    pred = pd.DataFrame()

    with open(args.input_dir + "/param.json", "r") as f:
        params = json.load(f)

    model = lgbm.LGBMRegressor(
        objective="regression",
        boosting_type="gbdt",
        n_estimators=3000,
        colsample_bytree=1,
        **params
    )

    n_splits = 4
    cv = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for i, (fit_idx, val_idx) in enumerate(cv.split(X_train, y_train)):

        X_fit = X_train.iloc[fit_idx]
        y_fit = y_train.iloc[fit_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        model.fit(
            X_fit,
            y_fit,
            eval_set=[(X_fit, y_fit), (X_val, y_val)],
            eval_names=("fit", "val"),
            eval_metric="l2",
            early_stopping_rounds=200,
            feature_name=X_fit.columns.tolist(),
            verbose=False,
        )

        pred["visitors"] = model.predict(X_test, num_iteration=model.best_iteration_)

    pred["visitors"] /= n_splits
    pred["visitors"] = np.expm1(pred["visitors"])
    pred.to_csv(args.output_dir + "/predict.csv", header=True, index=False)
