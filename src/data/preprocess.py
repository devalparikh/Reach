import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def create_dataframe(file_path):
    logging.info('Creating dataframe')
    print('Creating dataframe')
    return pd.read_csv(file_path)


def preprocess_dataframe(df, test_size, features, class_column):
    logging.info('Preprocessing')
    print('\n\n1. Preprocessing')

    print('\t1. Dropping Nans')
    df.dropna(inplace=True)

    # binary_cols = [col for col in df.columns if df[col].dtype not in [
    #     int, float] and df[col].nunique() == 2]

    # TODO: use binary_cols instead
    print('\t2. Label encoding')
    for col in df.columns:
        labelencoder = LabelEncoder()
        df[col] = labelencoder.fit_transform(df[col])

    X = df[features]
    y = df[class_column]

    print('\t3. Test train split')
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=test_size
    )

    print('\nX_train:\n' + X_train[:10].to_string())
    print('\ny_Train:\n' + y_train[:10].to_string())

    return X_train, X_test, y_train, y_test
