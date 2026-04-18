import joblib
import pandas as pd

# Load model
model = joblib.load('models/performance_model.pkl')

# Example new employee
new_employee = pd.DataFrame({
    'Age': [30],
    'Experience': [5],
    'Salary': [50000],
    'TrainingHours': [40],
    'ProjectsCompleted': [8]
})

prediction = model.predict(new_employee)

labels = {0: "Low", 1: "Medium", 2: "High"}

print("Predicted Performance:", labels[prediction[0]])