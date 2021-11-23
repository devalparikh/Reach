import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def _encode_labels(df, features, class_column):
    # binary_cols = [col for col in df.columns if df[col].dtype not in [
    #     int, float] and df[col].nunique() == 2]

    # TODO: use binary_cols instead
    for col in df.columns:
        labelencoder = LabelEncoder()
        df[col] = labelencoder.fit_transform(df[col])

    X = df[features]
    y = df[class_column]

    return X, y


def _tts(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=test_size
    )
    return X_train, X_test, y_train, y_test


def preprocess_dataframe(df, test_size, features, class_column):
    logging.info('Preprocessing')
    print('\n\n1. Preprocessing')

    print('\t1. Dropping Nans')
    df.dropna(inplace=True)

    print('\t2. Label encoding')
    X, y = _encode_labels(df, features, class_column)

    print('\t3. Test train split')
    X_train, X_test, y_train, y_test = _tts(X, y, test_size)

    return X_train, X_test, y_train, y_test
