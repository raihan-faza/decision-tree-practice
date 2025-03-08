import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# load dataset
def load_dataset():
    df = pd.read_csv("drug200.csv")
    return df


# checking info from dataset
def check_dataset_info(df: pd.DataFrame):
    # print info
    print(df.info())
    # print other info
    print(df.describe())
    # print shape
    print(df.shape)
    # cheking null value
    print(df.isna().sum())
    # checking class on all object dtype
    print(df.select_dtypes(include=["object"]).nunique())


# split dataset into train and validation
def split_dataset(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
):
    X = df.iloc[:, :-1]  # Select all columns except the last one
    y = df.iloc[:, -1]  # Select only the last column
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    return X_train, X_val, y_train, y_val


def encode_data(X_train, X_test):
    for col in X_train.select_dtypes(include=["object"]).columns:
        encoder = LabelEncoder()
        X_train[col] = encoder.fit_transform(X_train[col])
        X_test[col] = encoder.fit_transform(X_test[col])
    return X_train, X_test
