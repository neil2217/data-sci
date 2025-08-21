import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import zscore

# SNAP: Social network: Reddit Embedding Dataset [Using this dataset]
# https://snap.stanford.edu/data/web-RedditEmbeddings.html

# Load data
try:
    user_df = pd.read_csv('web-redditEmbeddings-users.csv', header=None)
    sub_df = pd.read_csv('web-redditEmbeddings-subreddits.csv', header=None)
    sub_df.rename(columns={0: 'subreddit_name'}, inplace=True)
except FileNotFoundError:
    print("Error: CSV files not found.")
    exit()

user_features = user_df.iloc[:, 1:]
sub_names = sub_df['subreddit_name']
sub_features = sub_df.iloc[:, 1:]

# Scale and remove outliers
scaler = StandardScaler()
user_scaled = scaler.fit_transform(user_features)
user_scaled_df = pd.DataFrame(user_scaled)
user_clean_mask = (np.abs(zscore(user_scaled_df)) < 3).all(axis=1)
user_clean = user_scaled_df[user_clean_mask]
print(f"Removed {len(user_features) - len(user_clean)} outliers; {len(user_clean)} users remain.")

# KMeans clustering
NUM_CLUSTERS = 5
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init='auto')
kmeans.fit(user_clean)
centroids = kmeans.cluster_centers_
print("Clustering complete.")

# Popular subreddits subset and scaling
popular = [
    'askreddit','funny','pics','todayilearned','worldnews','science','gaming','movies',
    'aww','music','videos','gifs','news','explainlikeimfive','books','television',
    'sports','nba','soccer','formula1','personalfinance','investing','wallstreetbets',
    'dataisbeautiful','mapporn','history','philosophy','programming','learnprogramming',
    'technology','gadgets','apple','android','politics','conservative','liberal',
    'food','cooking','fitness','travel','diy','woodworking','gardening','art'
]
pop_sub_df = sub_df[sub_df['subreddit_name'].isin(popular)].reset_index(drop=True)
pop_sub_scaled = scaler.transform(pop_sub_df.iloc[:, 1:])

# Interpret clusters via cosine similarity
TOP_N = 10
for i in range(NUM_CLUSTERS):
    centroid = centroids[i].reshape(1, -1)
    sims = cosine_similarity(centroid, pop_sub_scaled).flatten()
    top_idx = np.argsort(-sims)[:TOP_N]
    print(f"\nPersona for Cluster {i}:")
    print(pop_sub_df.loc[top_idx, 'subreddit_name'].to_string(index=False))

print("\nAnalysis complete.")