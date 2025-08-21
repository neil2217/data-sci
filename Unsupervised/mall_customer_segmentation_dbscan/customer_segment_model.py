import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
try:
    df = pd.read_csv('Mall_Customers.csv')
except FileNotFoundError:
    print("Error: 'Mall_Customers.csv' not found.")
    exit()

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
X_scaled = StandardScaler().fit_transform(X)

# Optimal DBSCAN parameters
min_pts = 4
nn = NearestNeighbors(n_neighbors=min_pts).fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
sorted_dist = np.sort(distances[:, min_pts-1])

plt.figure(figsize=(10,6))
plt.plot(sorted_dist)
plt.axhline(0.4, color='red', linestyle='--', label='Epsilon=0.4')
plt.title('K-Distance Elbow Plot')
plt.xlabel('Points sorted by distance')
plt.ylabel(f'Distance to {min_pts-1}-th Nearest Neighbor')
plt.legend()
plt.grid(True)
plt.show()

epsilon = 0.4
dbscan = DBSCAN(eps=epsilon, min_samples=min_pts)
clusters = dbscan.fit_predict(X_scaled)
df['Cluster'] = clusters

n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1)

print("="*30)
print(f"Epsilon: {epsilon}")
print(f"MinPts: {min_pts}")
print(f"Clusters found: {n_clusters}")
print(f"Noise points: {n_noise}")
print("="*30)

plt.figure(figsize=(12,8))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='deep', s=100)
plt.title('Customer Segments by DBSCAN')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()