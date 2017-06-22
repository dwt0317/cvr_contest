# -*- coding:utf-8 -*-
import copy
import pandas as pd
from util import constants
import math
import numpy as np


def compute_time_gap(time_lower, time_upper):
    return (time_upper / 10000 - time_lower / 10000) * 1440 \
    + ((time_upper % 10000) / 100 - (time_lower % 10000) / 100) * 60 \
    + (time_upper % 100 - time_lower % 100)


# deprecated
def build_user_click_time_gap(train_file, test_file, dir_path):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    train_data = train_data[['userID', 'label', 'clickTime', 'creativeID']]
    test_data = test_data[['userID', 'label', 'clickTime', 'creativeID']]
    train_data['first_click'] = 0
    test_data['first_click'] = 0
    train_data['time_delta'] = 0  # 与第一次点击的时间间隔
    test_data['time_delta'] = 0
    train_test_data = pd.concat([train_data, test_data], axis=0)

    # 处理train_test
    groups = train_test_data.groupby(['userID', 'creativeID'], as_index=False)
    groupnum = train_test_data[['userID', 'creativeID']].drop_duplicates().shape[0]
    print groupnum
    threshold = 60
    flag = 0
    header = ['instance', 'clickTime', 'continue_first_click', 'continue_not_first_click', 'time_delta']
    to_file = open(dir_path + 'train_test_time_delta_fea.csv', 'w')
    to_file.write(",".join(header) + '\n')

    for i, group in groups:
        continue_first_click_a = np.zeros(len(group)).astype(int)
        continue_not_first_click_a = np.zeros(len(group)).astype(int)
        time_delta_a = np.zeros(len(group)).astype(int)
        clickTime_a = np.asarray(group['clickTime'].tolist())
        # 不是单次点击的情况
        if len(group) != 1:
            time_delta_a[0] = compute_time_gap(clickTime_a[0], clickTime_a[1])
            if time_delta_a[0] < threshold:
                continue_first_click_a[0] = 1
            for i in range(1, len(group) - 1):
                last_gap = compute_time_gap(clickTime_a[i - 1], clickTime_a[i])
                if last_gap < threshold:
                    continue_not_first_click_a[i] = 1
                time_delta_a[i] = min(last_gap, clickTime_a[i], clickTime_a[i + 1])
            i = len(group) - 1
            last_gap = compute_time_gap(clickTime_a[i - 1], clickTime_a[i])
            if last_gap < threshold:
                continue_not_first_click_a[i] = 1
            time_delta_a[i] = last_gap
            time_gap_df = pd.DataFrame({'clickTime': pd.Series(clickTime_a),
                                        'continue_first_click': pd.Series(continue_first_click_a),
                                        'continue_not_first_click': pd.Series(continue_not_first_click_a),
                                        'time_delta': pd.Series(time_delta_a)}).set_index(group.index)
            values = time_gap_df.values
            indexes = list(time_gap_df.index)
            value2write = ""
            for j in xrange(len(values)):
                value2write += str(indexes[j]) + ',' + ",".join(str(v) for v in values[j]) + '\n'
            to_file.write(value2write)
        flag += 1
        if flag % 10000 == 0:
            print float(flag) / groupnum
    to_file.close()


def split_train_test(dir_path):
    train_test_fea = pd.read_csv(dir_path+'train_test_time_delta_fea.csv')
    print train_test_fea.head()
    train_fea = train_test_fea[train_test_fea['clickTime'] < 310000]
    test_fea = train_test_fea[train_test_fea['clickTime'] >= 310000]
    train_fea.sort('instance', inplace=True)
    test_fea.sort('instance', inplace=True)
    train_fea.to_csv(dir_path + 'train_time_delta_fea.csv', index=False)
    test_fea.to_csv(dir_path + 'predict_time_delta_fea.csv', index=False)


# 读取user time gap feature
def load_user_click_time_gap(dir_path, online):
    import copy
    time_delta_file = dir_path + 'train_time_delta_clean.csv'
    if online:
        time_delta_file = dir_path + 'train_time_delta2829_clean.csv'

    train_df = pd.read_csv(time_delta_file)
    train_values = np.asarray(train_df.values).astype(int)
    train_dict = {}
    for row in train_values:
        train_dict[int(row[4])] = copy.copy(list(row[:4]))

    predict_df = pd.read_csv(dir_path + 'predict_time_delta_clean.csv')
    predict_values = np.asarray(predict_df.values).astype(int)
    predict_dict = {}
    for row in predict_values:
        predict_dict[int(row[4])] = copy.copy(list(row[:4]))
    return train_dict, predict_dict


if __name__ == '__main__':
    dir_path = constants.project_path+"/dataset/custom/"
    train_file = "train_x_0"
    test_file = "test_x_0"

    # build_user_click_time_gap(constants.clean_train_path, constants.raw_test_path, dir_path)
    # split_train_test(dir_path)
    a, b = load_user_click_time_gap(train_file)
    print len(a), len(b)
    print a[:5, :]
    print b[:5, :]