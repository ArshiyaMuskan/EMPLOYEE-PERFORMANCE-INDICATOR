import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('data/cleaned_data.csv')

# Distribution
sns.countplot(x='Performance', data=data)
plt.title("Performance Distribution")
plt.savefig('outputs/performance_distribution.png')
plt.show()

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True)
plt.title("Correlation Matrix")
plt.savefig('outputs/correlation.png')
plt.show()

print("✅ EDA completed. Graphs saved in outputs/")