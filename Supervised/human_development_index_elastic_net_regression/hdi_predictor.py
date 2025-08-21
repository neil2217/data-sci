import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load and merge ---
try:
    df_index = pd.read_csv('1_HDI_Index_2022.csv', encoding='latin1')
    df_social = pd.read_csv('3_HDI_Social_Inequality_2022.csv', encoding='latin1')
    df_gender = pd.read_csv('4_HDI_Gender_Inequality_2022.csv', encoding='latin1')
except FileNotFoundError:
    print("CSV files missing.")
    exit()

data = pd.merge(df_index, df_social, on='Country').merge(df_gender, on='Country')

# --- Rename columns ---
mapping = {
    'Life expectancy at birth': 'Life expectancy at birth',
    'Gross national income (GNI) per capita': 'Gross national income (GNI) per capita',
    'Gini coefficient': 'Gini coefficient',
    'Overall loss (%)': 'Overall loss (%)',
    'Gender Inequality Index (GII)': 'Gender Inequality Index',
    'Maternal mortality ratio': 'Maternal mortality ratio',
    'Adolescent birth rate': 'Adolescent birth rate',
    'Share of seats in parliament': 'Share of seats in parliament',
    'Female secondary education': 'Female secondary education',
    'Male secondary education': 'Male secondary education',
    'Female labour force participation': 'Female labour force participation',
    'Male labour force participation': 'Male labour force participation'
}

rename_dict = {}
for clean, messy_start in mapping.items():
    for col in data.columns:
        if col.strip().startswith(messy_start):
            rename_dict[col] = clean
            break
data.rename(columns=rename_dict, inplace=True)

model_cols = list(mapping.keys())
model_data = data[model_cols].copy()

# Convert to numeric, remove commas, drop NaNs
for col in model_data:
    if model_data[col].dtype == 'object':
        model_data[col] = pd.to_numeric(model_data[col].str.replace(',', ''), errors='coerce')
model_data.dropna(inplace=True)

X = model_data.drop('Life expectancy at birth', axis=1)
y = model_data['Life expectancy at birth']

# --- Train/test split and scaling ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- ElasticNetCV training ---
model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, random_state=42, max_iter=10000)
model.fit(X_train_scaled, y_train)

print(f"Optimal alpha: {model.alpha_:.4f}")
print(f"Optimal l1_ratio: {model.l1_ratio_}")

# --- Evaluation ---
y_pred = model.predict(X_test_scaled)
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")

# --- Feature importance plot ---
coeffs = pd.Series(model.coef_, index=X.columns).sort_values()
plt.figure(figsize=(12,8))
sns.barplot(x=coeffs.values, y=coeffs.index, palette="vlag")
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Key Factors Influencing Life Expectancy')
plt.xlabel('Coefficient Value (Impact on Life Expectancy)')
plt.ylabel('Predictor Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')

print("\nFeature coefficients (negative to positive):")
print(coeffs)