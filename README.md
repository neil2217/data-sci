# Data Scientist – Machine Learning Algorithms

This repository contains a collection of **Machine Learning algorithms** organized by **Supervised** and **Unsupervised Learning** methods. Each project includes the implementation of an algorithm on a specific dataset.

---

## Setup and Installation

To run these projects, it's recommended to set up a virtual environment and install the required dependencies.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/neil2217/data-sci.git
    cd data-sci
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    This repository includes a `requirements.txt` file with all the necessary packages.
    ```bash
    python -m pip install -r requirements.txt
    ```

---

## Supervised Learning

Supervised learning uses labeled data to train models that can make predictions.

### 1. Classification

- **Decision Trees** – [Student Academic Stress Predictor](Supervised/student_stress_decision_tree)
- **Random Forest** – [Heart Disease Predictor](Supervised/heart_disease_random_forest)
- **SVM (Support Vector Machines)** – [Telco Customer Churn Predictor](Supervised/telco_customer_churn_svm)
- **KNN (K-Nearest Neighbors)** – [Cost of Living Tier Classifier](Supervised/cost_of_living_india_knn)
- **Logistic Regression** – [Online Shopper Purchase Intention Predictor](Supervised/online_shoppers_intention_logistic_regression)
- **Naive Bayes** – [Sarcasm in News Headlines Detector](Supervised/sarcasm_news_naive_bayes)
- **Gradient Boosting** – [Steam Game Recommender](Supervised/games_prediction_gradient_boost)

---

### 2. Regression

- **Linear Regression** – [Insurance Cost Predictor](Supervised/medical_cost_linear_regression)
- **Ridge Regression** – [Movie Revenue Predictor](Supervised/movies_ridge_regression)
- **Lasso Regression** – [House Price in India Predictor](Supervised/house_price_india_lasso_regression)
- **Elastic Net Regression** – [Life Expectancy Predictor (from HDI data)](Supervised/human_development_index_elastic_net_regression)

---

## Unsupervised Learning

Unsupervised learning uses unlabeled data to find hidden patterns and structures.

### 1. Clustering

- **K-Means** – [Reddit User Persona Clustering](Unsupervised/reddit_user_embeddings_k_means)
- **Hierarchical Clustering** – [Toughest Sport Hierarchical Analysis](Unsupervised/sports_skill_hierarchical_clustering)
- **DBSCAN** – [Mall Customer Segmentation using DBSCAN](Unsupervised/mall_customer_segmentation_dbscan)

### 2. Dimensionality Reduction

- **PCA (Principal Component Analysis)** – [Red Wine Quality PCA Analysis](Unsupervised/wine_quality_pca)