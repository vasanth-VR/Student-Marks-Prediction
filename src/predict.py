import joblib

# Load the trained model
model = joblib.load("models/model.pkl")

print("=== Student Marks Prediction ===")

# Get input from the user
hours = float(input("Enter study hours: "))

# Predict marks
predicted_marks = model.predict([[hours]])

print(f"\nPredicted Marks: {predicted_marks[0]:.2f}")