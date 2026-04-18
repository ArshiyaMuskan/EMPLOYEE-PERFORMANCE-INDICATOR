import pandas as pd
import numpy as np

np.random.seed(42)

n = 500

data = pd.DataFrame({
    'Age': np.random.randint(22, 60, n),
    'Experience': np.random.randint(1, 20, n),
    'Salary': np.random.randint(20000, 150000, n),
    'TrainingHours': np.random.randint(10, 100, n),
    'ProjectsCompleted': np.random.randint(1, 20, n)
})

# Create performance score
data['PerformanceScore'] = (
    0.3 * data['Experience'] +
    0.2 * data['TrainingHours'] +
    0.3 * data['ProjectsCompleted'] +
    np.random.normal(0, 5, n)
)

# Convert to categories
data['Performance'] = pd.qcut(data['PerformanceScore'], q=3, labels=['Low', 'Medium', 'High'])

# Save dataset
data.to_csv('data/employee_data.csv', index=False)

print("✅ Dataset created and saved in data/employee_data.csv")
print(data.head())