import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    wine_df = pd.read_csv('winequality-red.csv', sep=';')
except FileNotFoundError:
    print("Error: 'winequality-red.csv' not found.")
    exit()

X = wine_df.drop('quality', axis=1)
y = wine_df['quality']

X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
pca_df['quality'] = y.values

plt.figure(figsize=(12,8))
sns.scatterplot(x='PC1', y='PC2', hue='quality', data=pca_df, palette='viridis', alpha=0.8, edgecolor='k', s=60)
plt.title('2D PCA of Red Wine Quality Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Wine Quality')
plt.grid(True)
plt.savefig('wine_pca_visualization.png')
print("PCA visualization saved as 'wine_pca_visualization.png'")
plt.show()