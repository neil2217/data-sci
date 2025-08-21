import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

try:
    df = pd.read_csv('movies_metadata.csv', low_memory=False)
except FileNotFoundError:
    print("ERROR: 'movies_metadata.csv' not found.")
    exit()

features = ['budget', 'popularity', 'runtime']
target = 'revenue'
df_model = df[features + [target]].apply(pd.to_numeric, errors='coerce').dropna()
df_model = df_model[(df_model['budget'] > 1000) & (df_model['revenue'] > 1000)]

X = df_model[features]
y = df_model[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

y_pred = ridge.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: ${rmse:,.2f}")
print(f"R²: {r2:.4f} (~{r2:.1%} variance explained)")

print("Model Coefficients (scaled features):")
for feat, coef in zip(features, ridge.coef_):
    print(f" - {feat}: {coef:,.2f}")