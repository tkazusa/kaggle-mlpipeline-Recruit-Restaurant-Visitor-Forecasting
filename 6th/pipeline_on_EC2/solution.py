import glob
import time

import lightgbm as lgbm
import numpy as np
import pandas as pd
from scipy import optimize
from sklearn import metrics, model_selection


def load_air_visit(air_visit_path: str) -> pd.DataFrame:
    air_visit = pd.read_csv(air_visit_path)
    air_visit.index = pd.to_datetime(air_visit["visit_date"])
    air_visit = (
        air_visit.groupby("air_store_id")
        .apply(lambda g: g["visitors"].resample("1d").sum())
        .reset_index()
    )
    air_visit["visit_date"] = air_visit["visit_date"].dt.strftime("%Y-%m-%d")
    air_visit["was_nil"] = air_visit["visitors"].isnull()
    air_visit["visitors"].fillna(0, inplace=True)
    return air_visit


def load_date_info(date_info_path: str) -> pd.DataFrame:
    date_info = pd.read_csv(date_info_path)
    date_info.rename(
        columns={"holiday_flg": "is_holiday", "calendar_date": "visit_date"},
        inplace=True,
    )
    date_info["prev_day_is_holiday"] = date_info["is_holiday"].shift().fillna(0)
    date_info["next_day_is_holiday"] = date_info["is_holiday"].shift(-1).fillna(0)
    return date_info


def load_air_store_info(air_store_path: str) -> pd.DataFrame:
    air_store_info = pd.read_csv(air_store_path)
    return air_store_info


def load_submission(submission_path: str) -> pd.DataFrame:
    submission = pd.read_csv(submission_path)
    submission["air_store_id"] = submission["id"].str.slice(0, 20)
    submission["visit_date"] = submission["id"].str.slice(21)
    submission["is_test"] = True
    submission["visitors"] = np.nan
    submission["test_number"] = range(len(submission))
    submission.drop("id", axis="columns")
    return submission


def load_weather_data(weather_path: str) -> pd.DataFrame:
    weather_dfs = []

    for path in glob.glob(weather_path):
        weather_df = pd.read_csv(path)
        weather_df["station_id"] = path.split("/")[-1].rstrip(".csv")
        weather_dfs.append(weather_df)

    weather = pd.concat(weather_dfs, axis="rows")
    weather.rename(columns={"calendar_date": "visit_date"}, inplace=True)

    means = (
        weather.groupby("visit_date")[["avg_temperature", "precipitation"]]
        .mean()
        .reset_index()
    )
    means.rename(
        columns={
            "avg_temperature": "global_avg_temperature",
            "precipitation": "global_precipitation",
        },
        inplace=True,
    )

    weather = pd.merge(left=weather, right=means, on="visit_date", how="left")
    weather["avg_temperature"].fillna(weather["global_avg_temperature"], inplace=True)
    weather["precipitation"].fillna(weather["global_precipitation"], inplace=True)
    weather = weather[["visit_date", "avg_temperature", "precipitation", "station_id"]]
    return weather


def merge_train_test(
    air_visit: pd.DataFrame,
    submission: pd.DataFrame,
    date_info: pd.DataFrame,
    air_store_info: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    data = pd.concat((air_visit, submission))
    data["is_test"].fillna(False, inplace=True)
    data = pd.merge(left=data, right=date_info, on="visit_date", how="left")
    data = pd.merge(left=data, right=air_store_info, on="air_store_id", how="left")
    data = pd.merge(
        left=data, right=weather, on=["visit_date", "station_id"], how="left"
    )
    data["visitors"] = data["visitors"].astype(float)

    data["visit_date"] = pd.to_datetime(data["visit_date"])
    data.sort_values(["air_store_id", "visit_date"], inplace=True)
    data.index = data["visit_date"]
    return data


def replace_outliers_to_max_value(data: pd.DataFrame) -> pd.DataFrame:
    """Replace outliers of visitors to max value per each restaurant.
    It is assumed that all restaurant has different normal distribution of visitors, so values that lie out of confidence interval.
    """

    def _find_outliers(series):
        return (series - series.mean()) > 2.4 * series.std()

    def _cap_values(series):
        outliers = _find_outliers(series)
        max_val = series[~outliers].max()
        series[outliers] = max_val
        return series

    stores = data.groupby("air_store_id")
    data["is_outlier"] = stores.apply(lambda g: _find_outliers(g["visitors"])).values
    data["visitors_capped"] = stores.apply(lambda g: _cap_values(g["visitors"])).values
    data["visitors_capped_log1p"] = np.log1p(data["visitors_capped"])

    return data


def create_calender_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create calender features such like the day is weekend and the day of month."""
    data["is_weekend"] = data["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    data["day_of_month"] = data["visit_date"].dt.day
    return data


def create_ewm_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create exponentially weighted means features to capture the trend of timeseires."""

    def _calc_shifted_ewm(series, alpha, adjust=True):
        return series.shift().ewm(alpha=alpha, adjust=adjust).mean()

    def _find_best_signal(series, adjust=False, eps=10e-5):
        def _f(alpha):
            shifted_ewm = _calc_shifted_ewm(
                series=series, alpha=min(max(alpha, 0), 1), adjust=adjust
            )
            corr = np.mean(np.power(series - shifted_ewm, 2))
            return corr

        res = optimize.differential_evolution(func=_f, bounds=[(0 + eps, 1 - eps)])

        return _calc_shifted_ewm(series=series, alpha=res["x"][0], adjust=adjust)

    # Per store and the day of the week.
    roll = data.groupby(["air_store_id", "day_of_week"]).apply(
        lambda g: _find_best_signal(g["visitors_capped"])
    )
    ### merge でdata.index = data["visit_date"] はずすと、ここでコケる。
    data["optimized_ewm_by_air_store_id_&_day_of_week"] = roll.sort_index(
        level=["air_store_id", "visit_date"]
    ).values

    # Per store and week day or holiday.
    roll = data.groupby(["air_store_id", "is_weekend"]).apply(
        lambda g: _find_best_signal(g["visitors_capped"])
    )
    data["optimized_ewm_by_air_store_id_&_is_weekend"] = roll.sort_index(
        level=["air_store_id", "visit_date"]
    ).values

    # Per store and the day of the week for log1p visitors.
    roll = data.groupby(["air_store_id", "day_of_week"]).apply(
        lambda g: _find_best_signal(g["visitors_capped_log1p"])
    )
    data["optimized_ewm_log1p_by_air_store_id_&_day_of_week"] = roll.sort_index(
        level=["air_store_id", "visit_date"]
    ).values

    # Per store and week day or holiday for log1p visitors.
    roll = data.groupby(["air_store_id", "is_weekend"]).apply(
        lambda g: _find_best_signal(g["visitors_capped_log1p"])
    )
    data["optimized_ewm_log1p_by_air_store_id_&_is_weekend"] = roll.sort_index(
        level=["air_store_id", "visit_date"]
    ).values

    return data


def create_naive_rolling_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create rolling features."""

    def _extract_precedent_statistics(df, on, group_by):

        df.sort_values(group_by + ["visit_date"], inplace=True)

        groups = df.groupby(group_by, sort=False)

        stats = {"mean": [], "median": [], "std": [], "count": [], "max": [], "min": []}

        exp_alphas = [0.1, 0.25, 0.3, 0.5, 0.75]
        stats.update({"exp_{}_mean".format(alpha): [] for alpha in exp_alphas})

        for _, group in groups:

            shift = group[on].shift()
            roll = shift.rolling(window=len(group), min_periods=1)

            stats["mean"].extend(roll.mean())
            stats["median"].extend(roll.median())
            stats["std"].extend(roll.std())
            stats["count"].extend(roll.count())
            stats["max"].extend(roll.max())
            stats["min"].extend(roll.min())

            for alpha in exp_alphas:
                exp = shift.ewm(alpha=alpha, adjust=False)
                stats["exp_{}_mean".format(alpha)].extend(exp.mean())

        suffix = "_&_".join(group_by)

        for stat_name, values in stats.items():
            df["{}_{}_by_{}".format(on, stat_name, suffix)] = values

    data = data.reset_index(drop=True)

    _extract_precedent_statistics(
        df=data, on="visitors_capped", group_by=["air_store_id", "day_of_week"]
    )

    _extract_precedent_statistics(
        df=data, on="visitors_capped", group_by=["air_store_id", "is_weekend"]
    )

    _extract_precedent_statistics(
        df=data, on="visitors_capped", group_by=["air_store_id"]
    )

    _extract_precedent_statistics(
        df=data, on="visitors_capped_log1p", group_by=["air_store_id", "day_of_week"]
    )

    _extract_precedent_statistics(
        df=data, on="visitors_capped_log1p", group_by=["air_store_id", "is_weekend"]
    )

    _extract_precedent_statistics(
        df=data, on="visitors_capped_log1p", group_by=["air_store_id"]
    )

    return data


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


def perform_sanicy_check(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> None:
    assert X_train.isnull().sum().sum() == 0
    assert y_train.isnull().sum() == 0
    assert len(X_train) == len(y_train)
    assert X_test.isnull().sum().sum() == 0
    assert len(X_test) == 32019


def train_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    submission: pd.DataFrame,
    n_splits: int = 6,
) -> None:
    """Train a LightGBM regressor model and predict test set. The mdoel is evaluated utilizing KFold CV.
    Args:
        X_train (pd.DataFrame): Train features.
        y_train (pd.Series): Train target.
        X_test (pd.DataFrame): Test features.
        submission (pd.DataFrame):
    """
    np.random.seed(42)
    model = lgbm.LGBMRegressor(
        objective="regression",
        max_depth=5,
        num_leaves=5 ** 2 - 1,
        learning_rate=0.007,
        n_estimators=30000,
        min_child_samples=80,
        subsample=0.8,
        colsample_bytree=1,
        reg_alpha=0,
        reg_lambda=0,
        random_state=np.random.randint(10e6),
    )

    cv = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=42)
    val_scores = [0] * n_splits
    sub = submission["id"].to_frame()
    sub["visitors"] = 0

    feature_importances = pd.DataFrame(index=X_train.columns)

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

        val_scores[i] = np.sqrt(model.best_score_["val"]["l2"])
        sub["visitors"] += model.predict(X_test, num_iteration=model.best_iteration_)
        feature_importances[i] = model.feature_importances_

        print("Fold {} RMSLE: {:.5f}".format(i + 1, val_scores[i]))

    sub["visitors"] /= n_splits
    sub["visitors"] = np.expm1(sub["visitors"])

    val_mean = np.mean(val_scores)
    val_std = np.std(val_scores)

    print("Local RMSLE: {:.5f} (±{:.5f})".format(val_mean, val_std))


if __name__ == "__main__":
    print("solution started")
    start = time.time()
    # Path to datasets
    data_path = "../../data/"
    air_visit_path = data_path + "kaggle/air_visit_data.csv"
    date_info_path = data_path + "kaggle/date_info.csv"
    air_store_path = (
        data_path + "weather/air_store_info_with_nearest_active_station.csv"
    )
    submission_path = data_path + "kaggle/sample_submission.csv"
    weather_path = data_path + "weather/1-1-16_5-31-17_Weather/*.csv"

    # Load datasets
    air_visit = load_air_visit(air_visit_path=air_visit_path)
    date_info = load_date_info(date_info_path=date_info_path)
    air_store_info = load_air_store_info(air_store_path=air_store_path)
    weather = load_weather_data(weather_path=weather_path)
    submission = load_submission(submission_path=submission_path)

    # Merge train and test set to create features at once.
    data = merge_train_test(air_visit, submission, date_info, air_store_info, weather)
    print("completed")
    print(data.head())

    # Preprocessing and Feature Engineering
    data = replace_outliers_to_max_value(data=data)
    data = create_calender_features(data=data)
    data = create_ewm_features(data=data)
    data = create_naive_rolling_features(data=data)
    data = pd.get_dummies(data, columns=["day_of_week", "air_genre_name"])
    print("Preprocess and Feature Engineering completed")
    preprocess_end = time.time()
    print("elasped time {}sec".format(preprocess_end - start))

    # Split into train and test dataset.
    X_train, y_train, X_test = split_train_test(data=data)
    perform_sanicy_check(X_train=X_train, y_train=y_train, X_test=X_test)
    print("sanity check completed")

    # Train and predict
    train_predict(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        submission=submission,
        n_splits=6,
    )
    print("Train and predict completed")
    train_predict_end = time.time()
    print("elasped time {}sec".format(train_predict_end - preprocess_end))
