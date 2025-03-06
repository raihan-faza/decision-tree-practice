import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("drug200.csv")


# checking info from dataset
def check_dataset_info():
    print(df.info())
    print(df.describe())
    print(df.shape)
    print(df.isna().sum())


def split_dataset(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
):
    X = df.iloc[:, :-1]  # Select all columns except the last one
    y = df.iloc[:, -1]  # Select only the last column
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    return X_train, X_test, y_train, y_test


check_dataset_info()
