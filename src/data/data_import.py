import logging
import pandas as pd


def create_dataframe(file_path):
    logging.info('Creating dataframe')
    print('Creating dataframe')
    return pd.read_csv(file_path)
