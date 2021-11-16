import logging
import json

from data.preprocess import create_dataframe, preprocess_dataframe


class ReachModel:
    """
    A class used to represent an models

    ...

    Attributes
    ----------
    preproccessor : dict
        preproccessor
    model : dict
        model
    evaluation : dict
        evaluation

    Methods
    -------
    log_model_description()
        Logs the model structure
    """

    def __init__(self, raw_data_path, preproccessor, model, evaluation):
        self.df = create_dataframe(raw_data_path)
        self.preproccessor = preproccessor
        self.model = model
        self.evaluation = evaluation

    def preprocess(self):
        preprocess_dataframe(
            self.df,
            self.preproccessor['test_size'],
            self.preproccessor['selected_features'],
            self.preproccessor['class_column'],
        )

    def log_model_description(self):
        logging.info(vars(self))

        model_data = json.dumps(
            vars(self),
            indent=4,
            # sort_keys=True
        )
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
        'algorithm': 'multi_layer_perceptron',
        'epochs': 24,
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

    reach_model.preprocess()
