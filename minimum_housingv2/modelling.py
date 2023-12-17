import numpy as np
import pandas as pd
from joblib import load, dump
import unidecode
from final import PATH_FINAL_MODEL, FINAL_FEATURES_NAMES
import re
from utils import __cast_cat_type__, __cast_num_type__
from pre_processing import preprocessing_data, select_feature_data, filtering, fillna_missing


DEFAULT_INVALID_VALUE_RETURN = -1


class PreProcessing:
    def __init__(self, area,
                 pn,
                 duong,
                 ref_tinh_code,
                 ref_huyen_code,
                 ref_xa_code,
                 prj_name):
        self.area = area
        self.pn = pn
        self.duong = duong
        self.ref_tinh_code = ref_tinh_code
        self.ref_huyen_code = ref_huyen_code
        self.ref_xa_code = ref_xa_code
        self.prj_name = prj_name
        self.flag_valid_data = False

    def numeric_procesing(self):

        self.area = __cast_num_type__(self.area)
        self.pn = __cast_num_type__(self.pn)
        self.ref_tinh_code = __cast_num_type__(self.ref_tinh_code)
        self.ref_huyen_code = __cast_num_type__(self.ref_huyen_code)
        self.ref_xa_code = __cast_num_type__(self.ref_xa_code)
        return self

    def cat_processing(self):

        self.prj_name = __cast_cat_type__(self.prj_name)
        self.duong = __cast_cat_type__(self.duong)
        return self

    def check_valid_data(self):
        self.flag_valid_data = 25 <= self.area <= 300
        return self

    def processing_data(self):
        self.numeric_procesing()
        self.cat_processing()
        self.check_valid_data()
        return self


class Modelling:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.__load_model__()

    def __load_model__(self):
        self.model = load(self.model_path)
        return self

    def predict(self, **kwargs):
        data = PreProcessing(**kwargs).processing_data()
        if not data.flag_valid_data:
            return DEFAULT_INVALID_VALUE_RETURN
        else:
            feature = [data.area, data.pn, data.duong,
                       data.ref_tinh_code, data.ref_huyen_code,
                       data.ref_xa_code, data.prj_name]
            val = pd.DataFrame( columns=FINAL_FEATURES_NAMES)
            val.loc[0] = feature
            val = self.model.predict(val)
            val = int(val)
            val = max(val, 1e7)
            val = min(val, 3e8)
            return val

def convert_data(df, path_loc):
    df = preprocessing_data(df, path_loc)
    df = select_feature_data(df)
    df = filtering(df)
    df = fillna_missing(df)
    columns_select = ['area', 'pn', 'duong', 'ref_tinh_code', 'ref_huyen_code', 'ref_xa_code', 'prj_name']
    df = df[columns_select]
    return df

if __name__ == "__main__":
    from time import time
    s = time()
    path_loc =r'C:\Users\huuph\OneDrive\Documents\chungcu\Apartment-Price-Prediction\resources\loc.csv'
    model = Modelling(model_path=PATH_FINAL_MODEL)
    # vals = {"area": "50",
    #         "pn": 2,
    #         "duong": "trần bình",
    #         "ref_tinh_code": 233,
    #         "ref_huyen_code": 11,
    #         "ref_xa_code": 22,
    #         "prj_name": "vinhomes"}
    vals = {
        "price": 4500000000.0,
        "area": 106.0,
        "pn": 2,
        "toilet": 2,
        "date": "sunrise-city",
        "prj_name": "Đầy đủ",
        "noi_that": "Đông",
        "huong_nha": "Đông",
        "huong_ban_cong": "missing",
        "phap_ly": "Sổ đỏ/ Sổ hồng",
        "long": 10.73862361907959,
        "lat": 106.70059967041016,
        "duong": 'duong-nguyen-huu-tho',
        "xa": 'phuong-tan-hung-14',
        "huyen": 'Quận 7',
        "tinh": 'Hồ Chí Minh',
        "url": '',
        "source": 'bds'
    }
    X = pd.DataFrame.from_dict([vals], orient='columns')
    X = convert_data(X, path_loc)
    vals =X.to_dict(orient='records')[0]
    val = model.predict(**vals)
    s2 = time()
    print(val)
    print("total running time=", s2 - s)
    val = model.predict(**vals)
    s3 = time()
    print(val)
    print("total running time=", time() - s2)
