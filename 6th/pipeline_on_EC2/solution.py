import glob

import numpy as np
import pandas as pd
from scipy import optimize


def load_air_visit(air_visit_path: str):
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


def load_date_info(date_info_path: str):
    date_info = pd.read_csv(date_info_path)
    date_info.rename(
        columns={"holiday_flg": "is_holiday", "calendar_date": "visit_date"},
        inplace=True,
    )
    date_info["prev_day_is_holiday"] = date_info["is_holiday"].shift().fillna(0)
    date_info["next_day_is_holiday"] = date_info["is_holiday"].shift(-1).fillna(0)
    return date_info


def load_air_store_info(air_store_path):
    air_store_info = pd.read_csv(air_store_path)
    return air_store_info


def load_submission(submission_path):
    submission = pd.read_csv(submission_path)
    submission["air_store_id"] = submission["id"].str.slice(0, 20)
    submission["visit_date"] = submission["id"].str.slice(21)
    submission["is_test"] = True
    submission["visitors"] = np.nan
    submission["test_number"] = range(len(submission))
    submission.drop("id", axis="columns")
    return submission


def load_weather_data(weather_path):
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
):
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


def replace_outliers_to_max_value(data: pd.DataFrame):
    def _find_outliers(series):
        return (series - series.mean()) > 2.4 * series.std()

    def _cap_values(series):
        outliers = find_outliers(series)
        max_val = series[~outliers].max()
        series[outliers] = max_val
        return series

    stores = data.groupby("air_store_id")
    data["is_outlier"] = stores.apply(lambda g: _find_outliers(g["visitors"])).values
    data["visitors_capped"] = stores.apply(lambda g: _cap_values(g["visitors"])).values
    data["visitors_capped_log1p"] = np.log1p(data["visitors_capped"])

    return data


def create_month_features(data: pd.DataFrame):
    data["is_weekend"] = data["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    data["day_of_month"] = data["visit_date"].dt.day
    return data


def create_ewm_features(data: pd.DataFrame):
    def _calc_shifted_ewm(series, alpha, adjust=True):
        return series.shift().ewm(alpha=alpha, adjust=adjust).mean()

    def _find_best_signal(series, adjust=False, eps=10e-5):
        def _f(alpha):
            shifted_ewm = calc_shifted_ewm(
                series=series, alpha=min(max(alpha, 0), 1), adjust=adjust
            )
            corr = np.mean(np.power(series - shifted_ewm, 2))
            return corr

        res = optimize.differential_evolution(func=_f, bounds=[(0 + eps, 1 - eps)])

        return _calc_shifted_ewm(series=series, alpha=res["x"][0], adjust=adjust)

    roll = data.groupby(["air_store_id", "day_of_week"]).apply(
        lambda g: _find_best_signal(g["visitors_capped"])
    )
    data["optimized_ewm_by_air_store_id_&_day_of_week"] = roll.sort_index(
        level=["air_store_id", "visit_date"]
    ).values

    roll = data.groupby(["air_store_id", "is_weekend"]).apply(
        lambda g: _find_best_signal(g["visitors_capped"])
    )
    data["optimized_ewm_by_air_store_id_&_is_weekend"] = roll.sort_index(
        level=["air_store_id", "visit_date"]
    ).values

    roll = data.groupby(["air_store_id", "day_of_week"]).apply(
        lambda g: _find_best_signal(g["visitors_capped_log1p"])
    )
    data["optimized_ewm_log1p_by_air_store_id_&_day_of_week"] = roll.sort_index(
        level=["air_store_id", "visit_date"]
    ).values

    roll = data.groupby(["air_store_id", "is_weekend"]).apply(
        lambda g: _find_best_signal(g["visitors_capped_log1p"])
    )
    data["optimized_ewm_log1p_by_air_store_id_&_is_weekend"] = roll.sort_index(
        level=["air_store_id", "visit_date"]
    ).values

    return data


def create_naive_rolling_features(data: pd.DataFrame):
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


def split_train_test(data: pd.DataFrame):
    data["visitors_log1p"] = np.log1p(data["visitors"])
    train = data[
        (data["is_test"] == False)
        & (data["is_outlier"] == False)
        & (data["was_nil"] == False)
    ]
    test = data[data["is_test"]].sort_values("test_number")

    to_drop = [
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

    return X_train, X_test, y_train


def perform_sanicy_check(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame
):
    assert X_train.isnull().sum().sum() == 0
    assert y_train.isnull().sum() == 0
    assert len(X_train) == len(y_train)
    assert X_test.isnull().sum().sum() == 0
    assert len(X_test) == 32019


if __name__ == "__main__":
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
    data = pd.get_dummies(data, columns=["day_of_week", "air_genre_name"])
    print("completed")
    print(data.head())
