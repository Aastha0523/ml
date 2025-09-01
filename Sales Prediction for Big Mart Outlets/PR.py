import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PoissonRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.compose import TransformedTargetRegressor


data = pd.read_csv("train_v9rqX0R.csv")

def normalize_content(data):
    replace_map={
    'low fat' : 'Low Fat',
    'lf': 'Low Fat',
    'reg':'Regular',
    'LF': 'Low Fat',
    'Low fat': 'Low Fat'
    }
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace(replace_map)
    return data

data = normalize_content(data)

X = data.drop(columns=['Item_Outlet_Sales'])
Y = data['Item_Outlet_Sales']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=43)

cat_col = X_train.select_dtypes(include=['object','category']).columns
num_cols = X_train.select_dtypes(include=["number"]).columns

preprocessor = ColumnTransformer(
    transformers=[
    ('num', Pipeline([("imputer", SimpleImputer(strategy = "median")),
                      ("scaler", StandardScaler())]), num_cols),
    ('cat', Pipeline([("imputer", SimpleImputer(strategy = "most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown='ignore',sparse_output=False))]), cat_col),
])

pr = PoissonRegressor()
pr = Pipeline(steps=[
    ('preprocessor' ,preprocessor),
    ('regressor' ,PoissonRegressor(alpha=1.0, max_iter=1000))
])

pr.fit(X_train, Y_train)
Y_pred = pr.predict(X_test)

rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
print(rmse)

test_data = pd.read_csv("test_AbJTz2l.csv")
test_data = normalize_content(test_data)
predict = lr.predict(test_data)
selected_inputs = test_data[['Item_Identifier', 'Outlet_Identifier']]
output_df = selected_inputs.copy()
output_df['Item_Outlet_Sales'] = predict
output_df.to_csv('predictions_output.csv', index=False)