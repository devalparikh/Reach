import logging
import json


class Reach:
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
    logModelDescription()
        Logs the model structure
    """

    def __init__(self, preproccessor, model, evaluation):
        self.preproccessor = preproccessor
        self.model = model
        self.evaluation = evaluation

    def logModelDescription(self):
        model_data = json.dumps(
            vars(self),
            indent=4,
            # sort_keys=True
        )
        logging.info(vars(self))
        print(model_data)


if __name__ == '__main__':
    preprocess_config = {
        'train': 80,
        'test': 20,
        'cross_validation': {
            'k_folds': 4,
        },
        'selected_Features': ['Age', 'Price', 'Size']
    }

    model_config = {
        'algorithm': 'multi_layer_perceptron',
        'epochs': 24,
    }

    evaluation_config = {
        'metric': ['accuracy', 'f1']
    }

    model1 = Reach(preprocess_config, model_config, evaluation_config)
    model1.logModelDescription()
