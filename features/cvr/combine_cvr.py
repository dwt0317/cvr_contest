
# -*- coding:utf-8 -*-
import copy
import pandas as pd
from util import constants
import math

'''
用于生成组合cvr特征特征
'''

alpha = 135  # for smoothing
beta = 5085

left_day = 17
right_day = 24


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
    train_file_path = train_dir + "train_with_all_info_re.csv"
    total_df = pd.read_csv(train_file_path)
    partial_df = total_df[(total_df.clickTime >= left_day*10000) & (total_df.clickTime < right_day*10000)]
    pos_combine_dict = build_pos_combine_cvr(partial_df)
    app_combine_dict = build_app_combine_cvr(partial_df)
    # conn_combine_dict = build_conn_combine_cvr(partial_df)
    # cate_combine_dict = build_cate_combine_cvr(partial_df)
    # age_combine_dict = build_age_combine_cvr(partial_df)

    combine_cvr_dict = {}
    combine_cvr_dict['positionID'] = pos_combine_dict
    combine_cvr_dict['appID'] = app_combine_dict
    # combine_cvr_dict['connectionType'] = conn_combine_dict
    # combine_cvr_dict['appCategory'] = cate_combine_dict
    # combine_cvr_dict['age'] = age_combine_dict
    del total_df, partial_df
    return combine_cvr_dict


# 与positionID相关的组合cvr特征
def build_pos_combine_cvr(partial_df):
    positive_df = partial_df[partial_df.label == 1]
    pos_combine_dict = {}
    headers = ['appID', 'connectionType', 'campaignID', 'adID', 'creativeID', 'age', 'education', 'gender', 'haveBaby',
               'marriageStatus', 'hometown', 'residence', 'appCategory']
    combine_cvr_helper(partial_df, positive_df, headers, 'positionID', pos_combine_dict)
    print "Building pos combination cvr dict finished."
    return pos_combine_dict


# 与appID相关,
def build_app_combine_cvr(partial_df):
    positive_df = partial_df[partial_df.label == 1]
    app_combine_dict = {}
    headers = ['age', 'gender', 'education', 'connectionType']
    combine_cvr_helper(partial_df, positive_df, headers, 'appID', app_combine_dict)
    print "Building app combination cvr dict finished."
    return app_combine_dict


def build_conn_combine_cvr(partial_df):
    positive_df = partial_df[partial_df.label == 1]
    conn_combine_dict = {}
    headers = ['age', 'haveBaby', 'education']
    combine_cvr_helper(partial_df, positive_df, headers, 'connectionType', conn_combine_dict)
    print "Building connection combination cvr dict finished."
    return conn_combine_dict


def build_cate_combine_cvr(partial_df):
    positive_df = partial_df[partial_df.label == 1]
    cate_combine_dict = {}
    headers = ['age', 'haveBaby', 'education', 'residence']
    combine_cvr_helper(partial_df, positive_df, headers, 'appCategory', cate_combine_dict)
    print "Building category combination cvr dict finished."
    return cate_combine_dict


def build_age_combine_cvr(partial_df):
    positive_df = partial_df[partial_df.label == 1]
    age_combine_dict = {}
    headers = ['campaignID', 'adID', 'creativeID', 'appID']
    combine_cvr_helper(partial_df, positive_df, headers, 'age', age_combine_dict)
    print "Building category combination cvr dict finished."
    return age_combine_dict