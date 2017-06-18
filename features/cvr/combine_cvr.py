
# -*- coding:utf-8 -*-
import copy
import pandas as pd
from util import constants
import math
import cPickle as pickle
import datetime

'''
用于生成组合cvr特征特征
'''
# changed this to be same with init
alpha = 135  # for smoothing
beta = 5085

left_day = 17
right_day = 24

in_memory = False


def combine_cvr_helper(partial_df, positive_df, headers, another_header, combine_dict):
    for header in headers:
        groups = partial_df.groupby([header, another_header], as_index=False)
        clicks = groups.size()
        cv_groups = positive_df.groupby([header, another_header], as_index=False)
        cvs = cv_groups.size()
        combine_dict[header] = {}
        # header的attr值在前
        for k in clicks.keys():
            x = k[0]
            y = k[1]
            if x not in cvs or y not in cvs[x]:
                cv = 0
            else:
                cv = round(math.log(cvs[x][y], 2), 5)
            click = clicks[x][y]
            cvr = round((cv + alpha) / (float(click) + alpha + beta), 5)
            x = int(x)
            y = int(y)
            if x in combine_dict[header]:
                combine_dict[header][x][y] = [cv, cvr]
            else:
                combine_dict[header][x] = {y: [cv, cvr]}


def build_combine_cvr(train_dir):
    begin = datetime.datetime.now()
    train_file_path = train_dir + "train_with_all_info_clean.csv"
    if in_memory:
        total_df = pd.read_csv(train_file_path)
        positive_df = total_df[total_df.label == 1]
    else:
        total_df = None
        positive_df = None
    pos_combine_dict = build_pos_combine_cvr(total_df, positive_df, train_dir)
    app_combine_dict = build_app_combine_cvr(total_df, positive_df, train_dir)
    conn_combine_dict = build_conn_combine_cvr(total_df, positive_df, train_dir)
    # cate_combine_dict = build_cate_combine_cvr(partial_df, positive_df)
    # age_combine_dict = build_age_combine_cvr(partial_df, positive_df)
    # triple_combine_dict = build_triple_combine_cvr(partial_df, positive_df)

    combine_cvr_dict = {}
    combine_cvr_dict['positionID'] = pos_combine_dict
    combine_cvr_dict['appID'] = app_combine_dict
    combine_cvr_dict['connectionType'] = conn_combine_dict
    # combine_cvr_dict['appCategory'] = cate_combine_dict
    # combine_cvr_dict['age'] = age_combine_dict
    # combine_cvr_dict['triple'] = triple_combine_dict
    del total_df
    end = datetime.datetime.now()
    print "Combination cvr run time: " + str(end-begin)
    return combine_cvr_dict


# 与positionID相关的组合cvr特征
def build_pos_combine_cvr(partial_df, positive_df, train_dir):
    if in_memory:
        pos_combine_dict = {}
        headers = ['appID', 'connectionType', 'camgaignID', 'adID', 'creativeID', 'age', 'education', 'gender', 'haveBaby',
                   'marriageStatus']
        # , 'hometown', 'residence', 'appCategory'
        combine_cvr_helper(partial_df, positive_df, headers, 'positionID', pos_combine_dict)
        pickle.dump(pos_combine_dict,
                    open(constants.custom_path + '/features/pos_combine_dict.pkl', 'wb'))
    else:
        pos_combine_dict = pickle.load(
            open(train_dir + 'features/pos_combine_dict.pkl', 'rb'))
    print "Building pos combination cvr dict finished."
    return pos_combine_dict


# 与appID相关,
def build_app_combine_cvr(partial_df, positive_df, train_dir):
    if in_memory:
        app_combine_dict = {}
        headers = ['age', 'gender', 'education', 'connectionType']
        combine_cvr_helper(partial_df, positive_df, headers, 'appID', app_combine_dict)
        pickle.dump(app_combine_dict,
                    open(train_dir + 'features/app_combine_dict.pkl', 'wb'))
    else:
        app_combine_dict = pickle.load(
            open(train_dir + 'features/app_combine_dict.pkl', 'rb'))
    print "Building app combination cvr dict finished."
    return app_combine_dict


def build_conn_combine_cvr(partial_df, positive_df, train_dir):
    if in_memory:
        conn_combine_dict = {}
        headers = ['creativeID', 'camgaignID']
        combine_cvr_helper(partial_df, positive_df, headers, 'connectionType', conn_combine_dict)
        pickle.dump(conn_combine_dict,
                    open(train_dir + 'features/conn_combine_dict.pkl', 'wb'))
    else:
        conn_combine_dict = pickle.load(
            open(train_dir + 'features/conn_combine_dict.pkl', 'rb'))
    print "Building connection combination cvr dict finished."
    return conn_combine_dict


def build_cate_combine_cvr(partial_df, positive_df):
    cate_combine_dict = {}
    headers = ['age', 'haveBaby', 'education', 'residence']
    combine_cvr_helper(partial_df, positive_df, headers, 'appCategory', cate_combine_dict)
    print "Building category combination cvr dict finished."
    return cate_combine_dict


def build_age_combine_cvr(partial_df, positive_df):
    age_combine_dict = {}
    headers = ['marriageStatus', 'education', 'creativeID']
    combine_cvr_helper(partial_df, positive_df, headers, 'age', age_combine_dict)
    print "Building category combination cvr dict finished."
    return age_combine_dict


def build_triple_combine_cvr(partial_df, positive_df):

    groups = partial_df.groupby(['appID', 'positionID', 'connectionType'], as_index=False)
    clicks = groups.size()
    cv_groups = positive_df.groupby(['appID', 'positionID', 'connectionType'], as_index=False)
    cvs = cv_groups.size()
    combine_dict = {}
    for k in clicks.keys():
        x = k[0]
        y = k[1]
        z = k[2]
        if x not in cvs or y not in cvs[x] or z not in cvs[x][y]:
            cv = 0
        else:
            cv = round(math.log(cvs[x][y][z], 2), 5)
        click = clicks[x][y][z]
        cvr = round((cv + alpha) / (float(click) + alpha + beta), 5)
        x = int(x)
        y = int(y)
        z = int(z)
        if x in combine_dict:
            if y in combine_dict[x]:
                combine_dict[x][y][z] = [cv, cvr]
            else:
                combine_dict[x][y] = {z: [cv, cvr]}
        else:
            combine_dict[x] = {y: {z: [cv, cvr]}}
    return combine_dict


if __name__ == '__main__':
    combine_dict = build_combine_cvr(constants.custom_path+'/for_train/clean_id/')
