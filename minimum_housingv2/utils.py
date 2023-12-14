import unidecode
import re


def __cast_num_type__(x):
    try:
        return float(x)
    except:
        return -1


def __cast_cat_type__(x):
    x = str(x)
    x = unidecode.unidecode(x)
    x = x.upper().strip()
    x = re.sub(r"\s+", "-", x)
    if len(x) == 0 or len(x) > 50:
        return "MISSING"
    else:
        return x


def show_str_in_columns(str,
                        endline="\n",
                        seperator="|",
                        add_empty=" "):
    str = str.strip("\n")
    lines = str.split(endline)
    max_item_inline = 0
    dict_ = {i: 0 for i in range(100)}  # SẼ CÓ KHÔNG QUÁ 100 ITEM TRÊN 1 LINE
    for line in lines:
        vals = line.split(seperator)
        if len(vals) > max_item_inline:
            max_item_inline = len(vals)
        for i, val in enumerate(vals):
            dict_[i] = max(dict_[i], len(val))
    final_dict = {i: dict_[i] for i in range(max_item_inline)}
    max_val = len(final_dict) + sum(final_dict.values())
    between_line = "-" * max_val + endline
    final_str = between_line
    for line in lines:
        vals = line.split(seperator)
        vals_finals = []
        for i, val in enumerate(vals):
            add = final_dict[i] - len(val)
            if i + 1 == len(vals):
                this_val = add_empty * add + val
            else:
                this_val = val + add_empty * add
            vals_finals.append(this_val)
        vals_finals = f"{seperator}".join(vals_finals)
        final_str += vals_finals + endline
    final_str += between_line
    print(final_str)
    return final_str


def show_performance(my_pipeline, train, test, label_name, final_features_names):
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
    from src.utils import show_str_in_columns
    import numpy as np
    ALL = "__ALL__"
    keys = [ALL]
    keys += list(train.source.value_counts().index)
    strs = ""
    for key in keys:
        if key != ALL:
            feature_train = train[train.source == key]
            feature_test = test[test.source == key]
        else:
            feature_train = train.copy()
            feature_test = test.copy()
        y_train = feature_train[label_name]
        y_test = feature_test[label_name]
        X_train = feature_train[final_features_names]
        X_test = feature_test[final_features_names]
        if len(X_train) == 0:
            score = -1
            mape = -1
            r2 = -1
        else:
            pred_train = my_pipeline.predict(X_train)
            score = mean_absolute_error(y_train, pred_train)
            score = int(score)
            mape = mean_absolute_percentage_error(y_train, pred_train)
            r2 = r2_score(y_train, pred_train)
        strs += 'KEY={}| TRAIN| MAE={:,} | MAPE={} | R2={} | COUNT={:,}||'.format(key, score,
                                                                                  np.round(mape * 100, 2),
                                                                                  np.round(r2 * 100, 2), len(y_train))
        if len(X_test) == 0:
            score = -1
            mape = -1
            r2 = -1
        else:
            pred_test = my_pipeline.predict(X_test)
            score = mean_absolute_error(y_test, pred_test)
            score = int(score)
            mape = mean_absolute_percentage_error(y_test, pred_test)
            r2 = r2_score(y_test, pred_test)
        strs += 'TEST| MAE={:,} | MAPE={} | R2={}| COUNT={:,}'.format(score,
                                                                      np.round(mape * 100, 2),
                                                                      np.round(r2 * 100, 2), len(y_test))
        strs += "\n"

    show_str_in_columns(strs)


def __scoring__(model, data, key, final_feat ,label_col):
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
    import numpy as np
    X = data[final_feat]
    y = data[label_col]
    pred = model.predict(X)
    score = mean_absolute_error(y, pred)
    score = int(score)
    mape = mean_absolute_percentage_error(y, pred)
    r2 = r2_score(y, pred)

    strs = 'KEY={}| MAE={:,} | MAPE={} | R2={} | COUNT={:,}|\n'.format(key.upper(), score,
                                                                       np.round(mape * 100, 2),
                                                                       np.round(r2 * 100, 2), len(y))
    return strs
