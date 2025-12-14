import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# --- STEP 1: CREATE SIMPLE DATA ---
# Imagine we asked 5 people their experience and salary
data = {
    'YearsExperience': [1, 2, 3, 4, 5],
    'Salary':          [30000, 35000, 40000, 45000, 50000]
}
df = pd.DataFrame(data)

# --- STEP 2: TRAIN THE MODEL ---
# X is what we know (Experience), y is what we want to guess (Salary)
X = df[['YearsExperience']]
y = df['Salary']

model = LinearRegression()
model.fit(X, y)  # The computer "learns" the pattern here

# --- STEP 3: PREDICT ---
# Let's guess the salary for someone with 6 years of experience
years_to_predict = [[6]]
predicted_salary = model.predict(years_to_predict)

print(f"Predicted Salary for 6 years experience: ${predicted_salary[0]:.2f}")

# --- OPTIONAL: VISUALIZE IT ---
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', label="Prediction Line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()
