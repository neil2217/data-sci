import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- Load & preprocess data ---
df = pd.read_csv("House Price India.csv")
drop_cols = [c for c in ['id','Date','Unnamed: 0'] if c in df.columns]
df.drop(columns=drop_cols, inplace=True)
df.dropna(inplace=True)

X, y = df.drop("Price", axis=1), df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Pipeline: preprocess + LassoCV ---
num_feats = X.select_dtypes(include=np.number).columns.tolist()
cat_feats = X.select_dtypes(exclude=np.number).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_feats),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
], remainder='passthrough')

lasso_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LassoCV(cv=5, random_state=42, n_jobs=-1))
])

lasso_pipe.fit(X_train, y_train)
print(f"Training complete. Optimal α: {lasso_pipe['regressor'].alpha_:.4f}")

# --- Evaluation ---
y_pred = lasso_pipe.predict(X_test)
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: ₹{mean_absolute_error(y_test,y_pred):,.2f}")
print(f"RMSE: ₹{np.sqrt(mean_squared_error(y_test,y_pred)):,.2f}")

# --- Coefficients ---
lasso = lasso_pipe['regressor']
feat_names = num_feats
if cat_feats:
    feat_names += list(lasso_pipe['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_feats))

coefs = pd.Series(lasso.coef_, index=feat_names)
print(f"\nZeroed features: {len(coefs[coefs==0])}/{len(coefs)}")
print("\nTop + Influencers:\n", coefs.nlargest(10))
print("\nTop - Influencers:\n", coefs.nsmallest(10))