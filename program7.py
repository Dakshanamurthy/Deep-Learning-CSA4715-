import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


dataset = pd.read_csv("/content/drive/MyDrive/breastcancer.csv")


X = dataset.drop('Class', axis=1)
y = dataset['Class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
