import pandas as pd

# Load data
data = pd.read_csv('data/employee_data.csv')

# Check missing values
print("Missing values:\n", data.isnull().sum())

# Drop PerformanceScore (not needed for model)
data = data.drop(columns=['PerformanceScore'])

# Encode target
data['Performance'] = data['Performance'].map({
    'Low': 0,
    'Medium': 1,
    'High': 2
})

# Save cleaned data
data.to_csv('data/cleaned_data.csv', index=False)

print("✅ Preprocessing completed")
print(data.head())