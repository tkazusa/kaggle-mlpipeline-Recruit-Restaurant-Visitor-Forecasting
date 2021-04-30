# Kaggle Recruit Restaurant Visitor Forecasting コンペを題材に機械学習パイプラインの構築

Kaggle の [Recruit Restaurant Visitor Forecasting](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting) コンペの解法を題材に、機械学習におけるデータの前処理、学習、推論を自動化するパイプラインを AWS 上に構築します。

## 実行環境の構築

サンプルノートブックは Amazon SageMaker の Jupyter 環境であるノートブックインスタンス上で実行することを想定しています。

- ハンズオン環境は [コチラ](https://ap-northeast-1.console.aws.amazon.com/cloudformation/home?region=ap-northeast-1#/stacks/create/review?templateURL=https://recruit-pipeline-cfn-template.s3-ap-northeast-1.amazonaws.com/sagemaker-custom-resource.yaml&stackName=reqruit-ml-pipeline) のリンクから、AWS CloudFormation を活用してご自身のアカウントの東京リージョンに構築できます。「スタックを作成」ボタンを押して下さい。構築には10分程度かかります。
- 東京リージョン以外を活用したい、または、ハンズオン環境の詳細について知りたいという方は、[コチラ](https://github.com/tkazusa/kaggle-mlpipeline-titanic/blob/main/cfn-templates/sagemaker-custom-resource.yaml) の CloudFormation テンプレート をご確認下さい。

ハンズオンでは、[Kaggle](https://www.kaggle.com/) のデータセットを活用します。ダウンロードが可能なようにご自身のアカウントをご準備下さい。

## サンプルの構成

### データの準備

[dataprep.ipynb](https://github.com/tkazusa/kaggle-mlpipeline-Recruit-Restaurant-Visitor-Forecasting/blob/main/dataprep.ipynb) では Kaggle API を活用してデータのダウンロードを行い、Amazon S3 へデータをアップロードします。

### AWS のサービスを使ったサーバレスな前処理、学習、推論

[notebook.ipynb](https://github.com/tkazusa/kaggle-mlpipeline-Recruit-Restaurant-Visitor-Forecasting/blob/main/notebook.ipynb) では、AWS のマネージドな機械学習基盤サービスである、Amazon SageMaker を活用して、前処理や学習、推論を行います。データサイズが大きい、学習に時間がかかる、など Jupyter 環境では処理がしにくいような計算をそれぞれのジョブとしてサーバレスに実行するすることができます。

## 参考資料

- [8th place solution write-up](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/discussion/4916a6)
