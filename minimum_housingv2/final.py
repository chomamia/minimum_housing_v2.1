from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from minimum_housing.utils import (show_performance,
                       show_str_in_columns,
                       __cast_cat_type__,
                       __cast_num_type__,
                       __scoring__)

from joblib import load, dump
import os

FINAL_FEATURES_NAMES = ['area',
                        'pn',
                        'duong',
                        'ref_tinh_code',
                        'ref_huyen_code',
                        'ref_xa_code',
                        'prj_name',
                        'building_name']

NUMERIC_FEATURES = ['area', 'pn', 'ref_tinh_code', 'ref_huyen_code', 'ref_xa_code']

CATEGORICAL_FEATURES = ['duong',
                        'prj_name',
                        'building_name']

LABEL_NAME = "price_on_met"

LIGHTGBM_PARAMS = {"n_estimators": 10_000}


PATH_FINAL_DATA = r"D:\data\housing\house\processed\final_data_20220921_with_code.csv"
PATH_FINAL_MODEL = r"/minimum_housing\metadata\final_model_with_code.joblib"


def load_and_transform_data():

    def __transform__(data):
        df = data.copy()
        for col in NUMERIC_FEATURES:
            df[col] = df[col].map(lambda x: __cast_num_type__(x)).astype('float64')
        for col in CATEGORICAL_FEATURES:
            df[col] = df[col].map(lambda x: __cast_cat_type__(x))
        return df
    data = pd.read_csv(PATH_FINAL_DATA)
    data = __transform__(data)
    return data


def train_test_split(data, test_size=.1):
    from sklearn.model_selection import train_test_split
    data = data.sample(frac=1, random_state=12345)
    train, test = train_test_split(data, test_size=test_size, random_state=12345)
    return train, test


def modelling(mode):
    assert mode in ("check_performance", "get_final_model"), \
        f"mode={mode} not in 'check_performance', 'get_final_model'"
    data = load_and_transform_data()

    numerical_transformer = SimpleImputer(strategy='median')

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('imput', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])

    model = LGBMRegressor(**LIGHTGBM_PARAMS)

    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', model)
                                  ])
    if mode == "check_performance":
        print("running model to check  performance")
        train, test = train_test_split(data)
        X_train = train[FINAL_FEATURES_NAMES]
        y_train = train[LABEL_NAME]
        my_pipeline.fit(X_train, y_train)

        tt1 = __scoring__(model=my_pipeline, data=train, key="train", final_feat=FINAL_FEATURES_NAMES, label_col=LABEL_NAME)
        tt1 += __scoring__(model=my_pipeline, data=test, key="test", final_feat=FINAL_FEATURES_NAMES, label_col=LABEL_NAME)
        show_str_in_columns(tt1)
    elif mode == "get_final_model":
        print("run to save the output as final output")
        X_train = data[FINAL_FEATURES_NAMES]
        y_train = data[LABEL_NAME]
        my_pipeline.fit(X_train, y_train)
        tt = __scoring__(model=my_pipeline, data=data, key="train", final_feat=FINAL_FEATURES_NAMES, label_col=LABEL_NAME)
        show_str_in_columns(tt)
        print("save model at ", PATH_FINAL_MODEL)
        dump(my_pipeline, PATH_FINAL_MODEL)


if __name__ == "__main__":
    key1 = "get_final_model"
    key2 = "check_performance"
    modelling(key2)
    print()
    modelling(key1)
