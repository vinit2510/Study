import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


iris = datasets.load_wine()
X = iris.data
y = iris.target


random_seed = 42


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=random_seed)


classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=10000, random_state=random_seed),
    "Decision Tree": DecisionTreeClassifier(random_state=random_seed),
    "Random Forest": RandomForestClassifier(random_state=random_seed),
    "SVM": SVC(random_state=random_seed),
    "k-NN": KNeighborsClassifier()
}


for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"{name} Metrics:")
    print(f"  Accuracy: {accuracy:.2f}")
    print(f"  Precision: {precision:.2f}")
    print()


plt.figure(figsize=(10, 6))
plt.barh(list(classifiers.keys()), list([accuracy_score(y_test, clf.predict(X_test)) for clf in classifiers.values()]), color='skyblue')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
plt.title('Comparison of Supervised Learning Algorithms')
plt.xlim(0, 1.0)
plt.show()
