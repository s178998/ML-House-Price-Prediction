# 🏡 California House Price Prediction — Linear Regression Deep Dive

A complete machine learning project to predict **median house values in California** using **deep learning (TensorFlow)** and **regularization models (Ridge, Lasso)**.

---

## 🚀 What This Model Does

This model predicts housing prices using a real-world dataset: **California Housing** from `sklearn.datasets`. It uses both:

- A custom-built **neural network** trained with TensorFlow/Keras
- Regularized regression models (**Ridge**, **Lasso**) for comparison

---

## 🧰 Tools Used

| Purpose              | Library                |
|----------------------|------------------------|
| Deep learning        | TensorFlow / Keras     |
| Data manipulation    | Pandas, NumPy          |
| Visualization        | Matplotlib, Seaborn    |
| ML models & metrics  | Scikit-learn (Ridge, Lasso, GridSearchCV, R², MAE, MSE) |

---

## 📊 Model Results (Neural Network)

| Metric       | Value    |
|--------------|----------|
| R² Score     | ~0.85    |
| MAE          | ~0.30    |
| MSE          | ~0.21    |
| RMSE         | ~0.46    |

> ⚠️ Model is **still being tuned** — early stopping and dropout are implemented to reduce overfitting.

---

## 🧠 Deep Learning Architecture

```python
tf.keras.Sequential([
    tf.keras.layers.Input(shape=(n_features,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(1)  # Output layer for regression
])
