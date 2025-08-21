import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

try:
    df = pd.read_csv('insurance.csv')
except FileNotFoundError:
    print("Error: 'insurance.csv' not found.")
    exit()

print(df.head())

sns.lmplot(x='age', y='charges', hue='smoker', data=df, aspect=1.5, height=5)
plt.title('Age vs. Charges by Smoker Status')
plt.show()

df_enc = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

X = df_enc.drop('charges', axis=1)
y = df_enc['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

coeffs = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(f"Intercept: {model.intercept_}")
print(coeffs)