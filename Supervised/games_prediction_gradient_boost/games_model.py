import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# --- Load & preprocess ---
df = pd.read_csv("steam.csv")
df["genres"] = df["genres"].apply(lambda x: x.split(";"))
df["steamspy_tags"] = df["steamspy_tags"].apply(lambda x: x.split(";"))

total_ratings = df["positive_ratings"] + df["negative_ratings"]
df["rating_ratio"] = np.where(total_ratings > 0, df["positive_ratings"] / total_ratings, 0.5)

hard_kw = ['Difficult','Souls-like','Hard','Roguelike','Permadeath','Survival','Bullet Hell','Tactical','Grand Strategy']
easy_kw = ['Relaxing','Casual','Family Friendly','Walking Simulator','Hidden Object','Match 3','Clicker','Farming Sim']
df["difficulty"] = df["steamspy_tags"].apply(
    lambda tags: "Hard" if any(k in tags for k in hard_kw) else 
                 ("Easy" if any(k in tags for k in easy_kw) else "Normal")
)

# --- Features & target ---
features = df[["genres","difficulty","price","average_playtime"]]
target = df["rating_ratio"]

features_enc = pd.get_dummies(features, columns=["difficulty"], drop_first=True)
mlb = MultiLabelBinarizer()
genres_enc = pd.DataFrame(mlb.fit_transform(features_enc.pop("genres")), 
                          columns=mlb.classes_, index=features_enc.index)
X = pd.concat([features_enc, genres_enc], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, target, test_size=0.2, stratify=df["difficulty"], random_state=42
)

# --- Sample weights ---
train_diff = df.loc[X_train.index, "difficulty"]
weights = train_diff.map(lambda d: 30.0 if d=="Hard" else 1.5 if d=="Easy" else 1.0).to_numpy()

# --- Train & eval ---
model = lgb.LGBMRegressor(random_state=42)
model.fit(X_train, y_train, sample_weight=weights)
y_pred = model.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test,y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test,y_pred)):.4f}")

# --- Save model & encoder ---
joblib.dump(model, "steam_game_recommender.joblib")
joblib.dump(mlb, "mlb_genres_encoder.joblib")
print("Model and encoder saved.")

# --- Recommendation ---
def recommend_game(genre, difficulty, playtime):
    try:
        model = joblib.load("steam_game_recommender.joblib")
        mlb_genres = joblib.load("mlb_genres_encoder.joblib")
    except FileNotFoundError:
        return "Model files missing."

    candidates = df[df["difficulty"]==difficulty].copy()
    candidates = candidates[candidates["genres"].apply(lambda x: genre in x)]
    if playtime=="Short": candidates = candidates[candidates["average_playtime"]<600]
    elif playtime=="Medium": candidates = candidates[(candidates["average_playtime"]>=600)&(candidates["average_playtime"]<=3000)]
    else: candidates = candidates[candidates["average_playtime"]>3000]

    if candidates.empty: return "No game found."

    cand_feats = candidates[["genres","difficulty","price","average_playtime"]]
    cand_enc = pd.get_dummies(cand_feats, columns=["difficulty"])
    genres_enc = pd.DataFrame(mlb_genres.transform(cand_enc.pop("genres")),
                              columns=mlb_genres.classes_, index=cand_enc.index)
    X_cand = pd.concat([cand_enc, genres_enc], axis=1).reindex(columns=model.feature_name_, fill_value=0)

    candidates["pred_score"] = model.predict(X_cand)
    return candidates.sort_values("pred_score", ascending=False).iloc[0]["name"]

# --- Example ---
print("\nRecommended:", recommend_game("RPG","Hard","Long"))
print("Recommended:", recommend_game("Puzzle","Easy","Short"))
