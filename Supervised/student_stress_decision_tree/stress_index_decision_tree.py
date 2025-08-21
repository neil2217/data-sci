import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('academicStress.csv')
df.columns = ['Timestamp', 'AcademicStage', 'PeerPressure', 'HomePressure', 'StudyEnvironment',
              'CopingStrategy', 'BadHabits', 'AcademicCompetition', 'StressIndex']
df = df.drop('Timestamp', axis=1)

df_enc = pd.get_dummies(df, columns=['AcademicStage', 'StudyEnvironment', 'CopingStrategy', 'BadHabits'], drop_first=True)
X = df_enc.drop('StressIndex', axis=1)
y = df_enc['StressIndex']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

plt.figure(figsize=(25, 15))
plot_tree(dt_model, feature_names=X.columns, class_names=[str(c) for c in sorted(y.unique())],
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree for Predicting Student Stress (Max Depth=4)", fontsize=20)
plt.savefig("decision_tree_visualization.png")
plt.show()
print("Visualization saved as 'decision_tree_visualization.png'")