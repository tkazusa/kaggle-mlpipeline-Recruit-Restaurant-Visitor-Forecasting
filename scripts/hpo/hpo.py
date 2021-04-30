import argparse
import json
import re

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn import model_selection


def split_train_test(data: pd.DataFrame) -> pd.DataFrame:
    """Split data into train/test, and drop some features for training.
    Args:
        data (pd.DataFrame): The data which has whole features created.
    Returns:
        X_train (pd.DataFrame): Feature set for training.
        X_test (pd.DataFrame): Feature set for test.
        y_train (pd.Series): Values in the 'visitors_log1p' column for training set.
    """
    data["visitors_log1p"] = np.log1p(data["visitors"])
    train = data[
        (data["is_test"] == False)
        & (data["is_outlier"] == False)
        & (data["was_nil"] == False)
    ]
    test = data[data["is_test"]].sort_values("test_number")

    to_drop = [
        "id",
        "air_store_id",
        "is_test",
        "test_number",
        "visit_date",
        "was_nil",
        "is_outlier",
        "visitors_capped",
        "visitors",
        "air_area_name",
        "station_id",
        "latitude_str",
        "longitude_str",
        "station_latitude",
        "station_longitude",
        "station_vincenty",
        "station_great_circle",
        "visitors_capped_log1p",
    ]
    train = train.drop(to_drop, axis="columns")
    train = train.dropna()
    test = test.drop(to_drop, axis="columns")

    X_train = train.drop("visitors_log1p", axis="columns")
    X_test = test.drop("visitors_log1p", axis="columns")
    y_train = train["visitors_log1p"]

    return X_train, y_train, X_test


def objective(trial):

    # Todo: Download dataset from S3
    X_train = pd.read_csv(args.output_dir + "/X_train.csv")
    y_train = pd.read_csv(args.output_dir + "/y_train.csv")

    param = {
        "objective": "regression",
        "boosting_type": "gbdt",
        "n_estimators": 3000,
        "colsample_bytree": 1,
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    n_splits = 4
    val_scores = [0] * n_splits
    val_period_list = [i for i in range(1, n_splits)]
    X_train["period"] = np.arange(0, len(X_train)) // (len(X_train) // n_splits)
    X_train["period"] = np.clip(X_train["period"], 0, n_splits)

    for val_period in val_period_list:
        is_train = X_train["period"] < val_period
        is_val = X_train["period"] == val_period
        X_tr, X_vl = X_train[is_train], X_train[is_val]
        y_tr, y_vl = y_train[is_train], y_train[is_val]

        # LightGBMで学習+予測
        model = lgb.LGBMRegressor(**param)

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_train, y_train), (X_vl, y_vl)],
            eval_names=("train", "val"),
            eval_metric="l2",
            early_stopping_rounds=200,
            feature_name=X_train.columns.tolist(),
            verbose=False,
        )

        val_scores[val_period] = np.sqrt(model.best_score_["val"]["l2"])

    avg_val_score = np.array(val_scores).mean()

    return avg_val_score


if __name__ == "__main__":
    # Path to dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_trials", type=str, default=None)

    args, _ = parser.parse_known_args()

    data_path = args.input_dir + "/" + "data.csv"
    n_trials = int(args.n_traials)
    data = pd.read_csv(data_path)
    X_train, y_train, X_test = split_train_test(data)

    X_train.to_csv(args.output_dir + "/X_train.csv", header=True, index=False)
    y_train.to_csv(args.output_dir + "/y_train.csv", header=True, index=False)
    X_test.to_csv(args.output_dir + "/X_test.csv", header=True, index=False)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=7)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print(trial.params)

    # Upload the order flag as json file to S3 bucket.
    with open(args.output_dir + "/param.json", "w") as f:
        json.dump(trial.params, f, indent=4)
