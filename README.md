# House Prices Prediction

A small machine learning regression project for predicting house prices using the House Prices dataset from OpenML.

## Dataset

House Prices dataset from OpenML.

## Models

Two regression models were compared:

- Linear Regression
- HistGradientBoostingRegressor

## Preprocessing

### Numerical features
- Missing values are filled using median imputation.
- Features are standardized using StandardScaler.

### Categorical features
- Encoded using OneHotEncoder.
- Unknown categories are ignored.

## Results

| Model | CV Mean R² | CV Std |
|---|---:|---:|
| Linear Regression | 0.688 | 0.260 |
| HistGradientBoosting | 0.824 | 0.105 |

### HistGradientBoosting Test Performance

- Test R²: 0.884
- Test MAE: $16,966.38
- Test RMSE: $28,468.46

## Conclusion

HistGradientBoostingRegressor performed substantially better than Linear Regression on this dataset based on cross-validation and test R².
