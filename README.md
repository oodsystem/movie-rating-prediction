# Multiclass Movie Rating Prediction & K-Means Clustering

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oodsystem/movie-rating-prediction/blob/main/Movie_Rating_Prediction_CSE422.ipynb)

> **CSE422: Machine Learning Lab — BRAC University**
> S. M. Mohiuddin Khan Shiam · Saiyed Mubasshir Mahmood

---

## Overview

This project implements a complete end-to-end machine learning pipeline to predict a movie's rating category — **Excellent, Good, Average, or Poor** — from structured movie metadata. It covers the full ML workflow: data cleaning, EDA, feature engineering, supervised classification, unsupervised clustering, and model evaluation.

A secondary goal was to apply K-Means clustering to the same dataset (without labels) to see whether natural groupings emerge that align with the four rating categories.

---

## Dataset

A structured movie dataset with **1,200 instances** and **12 features**.

**Numerical features:** `DirectorPopularity`, `BudgetMillionUSD`, `RuntimeMinutes`, `ReleaseYear`, `NumMainActors`, `AvgActorPopularity`, `NumAwardsWon`, `MarketingSpendMillionUSD`

**Categorical features:** `Genre`, `HasFamousProducer`, `IsSequel`

**Target:** `RatingCategory` — one of `Excellent / Good / Average / Poor`

**Class distribution (training set):**

| Class | Count |
|-----------|-------|
| Good | 259 |
| Excellent | 243 |
| Average | 235 |
| Poor | 223 |

The dataset is **balanced** — max/min class ratio is well within acceptable range, so no oversampling was needed.

---

## Pipeline

### 1. Data Cleaning
- Duplicate check (0 duplicates found; removal step included defensively)
- Missing value imputation: **median** for numerical, **mode** for categorical — fitted on train set only to prevent leakage
- Outlier capping via IQR bounds on `BudgetMillionUSD` and `MarketingSpendMillionUSD`

### 2. Exploratory Data Analysis
- Class distribution plot to verify balance
- Correlation heatmap across all numeric features + encoded target
- Key finding: all numeric features showed very weak linear correlation with the target (max ≈ 0.04), indicating that non-linear models are more appropriate

### 3. Feature Engineering
- **One-hot encoding** for nominal features (e.g., `Genre`)
- **Binary encoding** for binary features (e.g., `IsSequel`)
- **Label encoding** for the target variable
- **StandardScaler** for all numerical features (inside `ColumnTransformer` to prevent leakage)
- Final feature count after encoding: **18**
- Train/test split: **80/20**, stratified by class

### 4. Supervised Learning
Five classifiers were trained and evaluated:

| Model | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| **Neural Network** | **0.2750**| **0.2725**| **0.2750**| **0.2708** |
| Logistic Regression | 0.2542 | 0.2503 | 0.2542 | 0.2471 |
| Decision Tree | 0.2458 | 0.2459 | 0.2458 | 0.2434 |
| Naive Bayes | 0.2458 | 0.2378 | 0.2458 | 0.2295 |
| KNN (k=5) | 0.2333 | 0.2374 | 0.2333 | 0.2335 |

> **Note on results:** All models performed near the random baseline for a 4-class problem (~25%). This is consistent with the near-zero feature-target correlations found in EDA. The dataset's rating labels appear to depend on complex non-linear interactions not well captured by these features alone — a realistic outcome in many real-world ML tasks. The neural network architecture used: `Dense(64, ReLU) → Dropout(0.2) → Dense(32, ReLU) → Dense(4, Softmax)`, trained with Adam for 50 epochs.

### 5. Unsupervised Learning — K-Means Clustering
- Elbow method (WCSS over k=1–10) used to determine optimal cluster count
- K set to **4** to align with the four known rating categories
- Clusters visualized in 2D using **PCA** dimensionality reduction
- Result: clusters did not cleanly separate by rating category, consistent with the weak feature signal found in supervised learning

---

## Key Takeaways

- Weak linear feature-target correlation is a strong signal to try non-linear or ensemble models (Random Forest, XGBoost, Gradient Boosting) in future work
- Proper train-test split **before** any preprocessing is essential to prevent data leakage — this pipeline follows that correctly
- K-Means confirmed that the data does not naturally partition into four clean groups, suggesting the rating labels may be noisy or dependent on features not present in the dataset
- The full pipeline demonstrates solid ML engineering practices regardless of final accuracy

---

## How to Run

1. Click the **Open in Colab** badge above
2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Upload the dataset to: `MyDrive/CSE422_Project/movie_ratings_dataset.csv`
4. Run all cells sequentially

### Dependencies

All packages are available on Colab by default:

| Package | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | ML models, preprocessing, metrics |
| `tensorflow` / `keras` | Neural network |
| `matplotlib`, `seaborn` | Visualization |

---

## Future Improvements

- Try ensemble methods (Random Forest, XGBoost, LightGBM) which handle non-linear patterns better
- Feature engineering — add interaction terms or polynomial features
- Hyperparameter tuning via GridSearchCV or Optuna
- Investigate label quality — noisy or inconsistently labeled data could explain near-random performance
- Experiment with deeper neural network architectures or regularization strategies

---

## Authors

| Name | Student ID |
|------|-----------|
| S. M. Mohiuddin Khan Shiam | 24341255 |
| Saiyed Mubasshir Mahmood | 22101500 |

**Course:** CSE422 — Machine Learning Lab, BRAC University
**Submission:** December 2025

---

## License

This project is for academic and portfolio purposes only.
