import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------
# Load Dataset
# -------------------------------
data = pd.read_csv("data/student_scores.csv")

# Display first 5 rows
print("Dataset Preview:")
print(data.head())

# -------------------------------
# Select Features and Target
# -------------------------------
X = data[["Hours"]]
y = data["Marks"]

# -------------------------------
# Split Dataset
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------
# Train Model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# Predict
# -------------------------------
predictions = model.predict(X_test)

# -------------------------------
# Evaluate Model
# -------------------------------
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("----------------------------")
print(f"Mean Absolute Error : {mae:.2f}")
print(f"R² Score            : {r2:.2f}")

# -------------------------------
# Save Model
# -------------------------------
joblib.dump(model, "models/model.pkl")

metrics = {
    "mae": mae,
    "r2": r2
}

joblib.dump(metrics, "models/metrics.pkl")

print("\nModel and metrics saved successfully!")