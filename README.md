# kaggle-mlpipeline-Recruit-Restaurant-Visitor-Forecasting

## 実行時間の計測
### 1st solution の前処理にかかる時間
- 実行環境のメモ
    - Python 3.8
    - EC2 t3.2xlarge
    - Ubuntu 20.04 DL AMI


- 実行内容
    - 1 time window: 85 sec, Data shape: (28429, 219)
        - start_date = "2017-03-12"
        - end_date = "2017-04-19"
        - n_days = 35