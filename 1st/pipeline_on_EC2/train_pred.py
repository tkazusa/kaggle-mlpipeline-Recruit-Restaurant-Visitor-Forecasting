import datetime
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

data_path = "../data/"
submission = pd.read_csv(data_path + "sample_submission.csv")

submission["visit_date"] = submission["id"].str[-10:]
submission["store_id"] = submission["id"].str[:-11]

train_feat = pd.read_csv("train_feat_multi.csv")
test_feat = pd.read_csv("test_feat.csv")
exclude_cols = [
    "id",
    "store_id",
    "visit_date",
    "end_date",
    "air_area_name",
    "visitors",
    "month",
]

predictors = [feat for feat in test_feat.columns if feat not in exclude_cols]

params = {
    "learning_rate": 0.02,
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "sub_feature": 0.7,
    "num_leaves": 60,
    "min_data": 100,
    "min_hessian": 1,
    "verbose": 1,
}

t0 = time.time()
lgb_train = lgb.Dataset(train_feat[predictors], train_feat["visitors"])
lgb_test = lgb.Dataset(test_feat[predictors], test_feat["visitors"])

gbm = lgb.train(params, lgb_train, 2300)
train_time = time.time()
print("Training : {}sec".format(train_time - t0))
pred = gbm.predict(test_feat[predictors])

pred_time = time.time()
print("Inference : {}sec".format(pred_time - train_time))
subm = pd.DataFrame(
    {"id": test_feat.store_id + "_" + test_feat.visit_date, "visitors": np.expm1(pred)}
)
subm = submission[["id"]].merge(subm, on="id", how="left").fillna(0)
subm.to_csv(
    "sub{}.csv".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")),
    index=False,
    float_format="%.4f",
)
