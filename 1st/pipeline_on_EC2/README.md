# Train and Prediction Pipeline on EC2

## Testing Environment

- Amazon EC2 m5.2xlarge
- Ubuntu 20.04 LTS
- Python3.8

## Test Procedure

### Preprocess using single window for label data.

```
> python measure preprocess_single.py
Data Loading Start
Feature Engineering Start
...
...
Feature Engineering Completed
Data Size: (28492, 224)
Time elasped: 78 sec
```

### Proprocess using multi window for label data.

```
> python measure preprocess_single.py
Data Loading Start
Feature Engineering Start
...
...
Feature Engineering Completed
Data Size: (28492, 224)
Time elasped: 78 sec
```
  
### Train and Predictiona

```
> python train_pred.py
```