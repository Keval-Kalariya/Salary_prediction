import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. CREATE DATA MANUALY (Instead of CSV)

data = {
    'YearsExperience': [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7,
                        3.9, 4.0, 4.0, 4.1, 4.5, 4.9, 5.1, 5.3, 5.9, 6.0,
                        6.8, 7.1, 7.9, 8.2, 8.7, 9.0, 9.5, 9.6, 10.3, 10.5],
    'Salary': [39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445, 57189,
               63218, 55794, 56957, 57081, 61111, 67938, 66029, 83088, 81363, 93940,
               91738, 98273, 101302, 113812, 109431, 105582, 116969, 112635, 122391, 121872]
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# 2. DATA CLEANING

# Even with manual data, we include this step to match your project description
print("Checking for missing values:\n", df.isnull().sum())
df = df.drop_duplicates()  # Remove duplicate rows if any

# 3. EDA & VISUALIZATION

plt.figure(figsize=(8, 5))
plt.scatter(df['YearsExperience'], df['Salary'],
            color='blue', label='Data Points')
plt.title('Years of Experience vs Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

# 4. SPLIT DATA

X = df[['YearsExperience']]
y = df['Salary']

# Split: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 5. TRAIN MODELS (Linear Regression & Random Forest)

# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Model 2: Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# 6. EVALUATION METRICS (MAE, RMSE, R2)


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"--- {name} Results ---")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}\n")


# Print results
print("\n=== MODEL PERFORMANCE REPORT ===\n")
evaluate("Linear Regression", y_test, lr_pred)
evaluate("Random Forest", y_test, rf_pred)

# 7. LIVE TEST (Optional)
# Let's predict salary for someone with 5.5 years exp


new_pred = lr_model.predict([[5.5]])
print(f"Prediction for 5.5 Years Experience: ${new_pred[0]:,.2f}")
