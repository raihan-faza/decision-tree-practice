import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from data_process import encode_data, load_dataset, split_dataset

df = load_dataset()
X_train, X_test, y_train, y_test = split_dataset(df=df, test_size=0.2, random_state=1)
X_train, X_test = encode_data(X_train, X_test)
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))
