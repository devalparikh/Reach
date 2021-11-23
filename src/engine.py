import logging
import json

from data.data_import import create_dataframe
from features.preprocess import preprocess_dataframe
from models.train import train


class ReachModel:
    """
    A class used to represent an models

    ...

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, raw_data_path, preproccessor, model, evaluation):
        self.df = create_dataframe(raw_data_path)
        self.preproccessor = preproccessor
        self.model = model
        self.evaluation = evaluation

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def preprocess(self):
        X_train, X_test, y_train, y_test = preprocess_dataframe(
            self.df,
            self.preproccessor['test_size'],
            self.preproccessor['selected_features'],
            self.preproccessor['class_column'],
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train(self):
        model = train(self.model, self.X_train, self.y_train)
        print()

    def log_model_config(self):
        model_data = json.dumps(
            {**self.preproccessor,
             **self.model,
             **self.evaluation},
            indent=4,
            # sort_keys=True
        )

        logging.info(model_data)
        print(model_data)


if __name__ == '__main__':

    raw_data_path = './src/data/music_genre_data.csv'

    preprocess_config = {
        'test_size': 25,
        'cross_validation': {
            'k_folds': 4,
        },
        'selected_features': ['artist_name', 'key', 'instrumentalness'],
        'class_column': ['music_genre']
    }

    model_config = {
        # 'algorithm': 'multi_layer_perceptron',
        # 'epochs': 24,
        'algorithm': 'knn',
        'k': 3

    }

    evaluation_config = {
        'metric': ['accuracy', 'f1']
    }

    reach_model = ReachModel(
        raw_data_path,
        preprocess_config,
        model_config,
        evaluation_config
    )

    reach_model.log_model_config()
    reach_model.preprocess()
    print('\nX_train:\n' + reach_model.X_train[:10].to_string())
    print('\ny_Train:\n' + reach_model.y_train[:10].to_string())
    reach_model.train()
    # reach_model.evaluate()
