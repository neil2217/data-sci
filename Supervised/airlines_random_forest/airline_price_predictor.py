import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load and preprocess data
df = pd.read_csv('airlines_flights_data.csv').drop(columns=['index', 'flight'])
X = df.drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
cat_cols = X_train.select_dtypes('object').columns.tolist()
encoder = ce.TargetEncoder(cols=cat_cols).fit(X_train, y_train)
X_train_enc = encoder.transform(X_train)
X_test_enc = encoder.transform(X_test)

# Train model
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_enc, y_train)

# Evaluate
y_pred = rf.predict(X_test_enc)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

# Feature importance
feat_imp = pd.DataFrame({'feature': X_train_enc.columns, 'importance': rf.feature_importances_}) \
    .sort_values('importance', ascending=False).head(15)

sns.barplot(x='importance', y='feature', data=feat_imp, hue='feature',palette='viridis', legend=False)
plt.title('Top 15 Most Important Features in Predicting Flight Price')
plt.tight_layout()
plt.show()