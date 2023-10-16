import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
data = pd.read_csv("medicle.csv", names=names)

# Data Preprocessing
X = data.drop(columns=['target'])
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Visualizations (you can keep this part)
plt.hist(data['age'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()

plt.hist(data['chol'], bins=20, color='orange', edgecolor='black')
plt.xlabel('Cholesterol Level')
plt.ylabel('Frequency')
plt.title('Cholesterol Level Distribution')
plt.show()

plt.scatter(data['age'], data['chol'], color='green', alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Cholesterol Level')
plt.title('Age vs. Cholesterol Level')
plt.show()

# Bayesian Network Inference Function
def bayesian_network_inference(patient_data):
    if patient_data['exang'] == 0:
        return 0  # No heart disease
    else:
        return 1  # Heart disease

# Inference on the Test Set
predicted = []
for _, patient_data in X_test.iterrows():
    result = bayesian_network_inference(patient_data)
    predicted.append(result)

# Evaluation
accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted, average='weighted')
recall = recall_score(y_test, predicted, average='weighted')
f1 = f1_score(y_test, predicted, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
