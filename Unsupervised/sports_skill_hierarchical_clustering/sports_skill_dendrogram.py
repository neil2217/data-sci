import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

try:
    df = pd.read_csv('toughestsport.csv')
except FileNotFoundError:
    print("Error: 'toughestsport.csv' not found.")
    exit()

df.set_index('SPORT', inplace=True)
features = ['Endurance', 'Strength', 'Power', 'Speed', 'Agility',
            'Flexibility', 'Nerve', 'Durability', 'Hand-eye coordination', 'Analytical Aptitude']
X = df[features]

X_scaled = StandardScaler().fit_transform(X)
linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(15,10))
dendrogram(linked, orientation='top', labels=df.index, distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering of Sports (Dendrogram)')
plt.ylabel('Distance (Ward Linkage)')
plt.xlabel('Sport')
plt.tight_layout()
plt.show()