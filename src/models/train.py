from numpy import mod
from sklearn.neighbors import KNeighborsClassifier


def _knn_train(model_config):
    print('\t1. KNN Training')
    k = model_config['k']
    model = KNeighborsClassifier(n_neighbors=k)
    return model


def _mlp_train():
    print()


algorithms = {
    'knn': _knn_train,
    'mlp': _mlp_train,
}

# TODO: add automated hyperparameter optimization


def train(model_config, X_train, y_train):
    print('\n2. Training')
    algorithm = model_config['algorithm']
    model = algorithms[algorithm](model_config)

    print(model)
    model.fit(X_train, y_train.values.ravel())
    # print(model.predict([[214, 4, 0]]))
    return model
