from sklearn.metrics import mean_absolute_error,root_mean_squared_error
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate,train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression



housing = fetch_openml(name='house_prices', version=1 , as_frame=True)

X = housing.data
y = housing.target

numerical_selector = selector(dtype_exclude=object)
numerical_columns = numerical_selector(X)

categorical_selector = selector(dtype_include=object)
categorical_columns = categorical_selector(X)

numerical_preprocess = make_pipeline(SimpleImputer(strategy='median'),StandardScaler())
categorical_preprocess = OneHotEncoder(handle_unknown='ignore',sparse_output=False)

preprocess1 = make_column_transformer((numerical_preprocess,numerical_columns),
                                     (categorical_preprocess,categorical_columns))


X_train , X_test , y_train , y_test = train_test_split(X,y, test_size=0.25,random_state=42)

model1 = make_pipeline(preprocess1,LinearRegression()) 

cv_res1 = cross_validate(model1,X_train,y_train)
scores1 = cv_res1['test_score']

print(f"Linear Regression:\nmean= {scores1.mean():.3f}, std= {scores1.std():.3f}")


preprocess2 = make_column_transformer((numerical_preprocess,numerical_columns),
                                      (categorical_preprocess,categorical_columns))

model2 = make_pipeline(preprocess2,HistGradientBoostingRegressor())
cv_res2 = cross_validate(model2,X_train,y_train)
scores2 = cv_res2['test_score']

print(f"HistGradientBoosting:\nmean= {scores2.mean():.3f}, std= {scores2.std():.3f}")


model2.fit(X_train,y_train)

y_pred = model2.predict(X_test)

test_r2 = model2.score(X_test,y_test) 
test_mae = mean_absolute_error(y_test,y_pred)
test_rmse = root_mean_squared_error(y_test, y_pred)


print(f"Test R²: {test_r2:.3f}")
print(f"Test MAE: ${test_mae:,.2f}") #$16,966.38
print(f"Test RMSE: ${test_rmse:,.2f}") #$28,468.46
