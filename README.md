# Predicting California House Prices Using Deep Learning 🏡📊

This project is my implementation of a deep learning model that predicts median house values in California using the **California Housing dataset**. I built this to sharpen my skills in machine learning, particularly with **TensorFlow**, and to get hands-on experience with a real-world regression problem from start to finish.

## 🔍 Problem I’m Solving

The goal here was to predict how much houses cost in different neighborhoods in California based on features like income, location, population, and housing stats. It’s a great example of how machine learning can be used for real estate forecasting and data-driven investment planning.

## 🛠️ Tools & Libraries I Used

- Python
- TensorFlow
- Pandas & NumPy
- Seaborn & Matplotlib
- Scikit-learn

## 🧠 Model Breakdown

- **Model Type**: Deep Neural Network for regression
- **Features Used**: 8 numerical values (like median income, latitude, rooms per household, etc.)
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Evaluation**: Mean Absolute Error (MAE) and R² Score

## 📈 Results

- Reached over **85% R² score** on the test set
- Visualized the model’s predictions vs actual values
- Improved the model using **early stopping** and **dropout layers** to reduce overfitting

## 🧪 How to Run This

If you want to run this yourself, clone the repo and install the required libraries:

```bash
pip install -r requirements.txt
