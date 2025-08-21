import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
data = data.drop('customerID', axis=1)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

X = pd.get_dummies(data.drop('Churn', axis=1), drop_first=True)
y = data['Churn'].map({'Yes': 1, 'No': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(kernel='rbf', C=1.0, random_state=42, class_weight='balanced')
svm.fit(X_train_scaled, y_train)
y_pred = svm.predict(X_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))