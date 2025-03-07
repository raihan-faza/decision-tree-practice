import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from data_process import load_dataset, split_dataset

df = load_dataset()
X_train, X_test, y_train, y_test = split_dataset(df=df, test_size=0.2, random_state=1)
classifier = DecisionTreeClassifier()
