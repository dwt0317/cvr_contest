# -*- coding:utf-8 -*-
import copy
import pandas as pd
from util import constants
import math
import numpy as np


def build_last_creative_feature():
    train_data = pd.read_csv(constants.clean_train_path)
    test_data = pd.read_csv(constants.raw_test_path)
    train_data = train_data[['userID', 'creativeID', 'label', 'clickTime']]
    test_data = test_data[['userID', 'creativeID', 'label', 'clickTime']]
    train_test_data = pd.concat([train_data, test_data], axis=0)
    print train_test_data.head()
    train_test_data.drop_duplicates(inplace=True)
    groups = train_test_data.groupby(['userID'], as_index=False)
    header = ['instance', 'clickTime', 'different_last_creative']
    to_file = open(constants.project_path+'/dataset/custom/' + 'train_test_last_click_fea.csv', 'w')
    to_file.write(",".join(header) + '\n')
    count = 0
    positive_count = 0
    # print len(groups)
    for i, group in groups:
        # 不是单次点击的情况
        n = len(group)
        if n > 2:
            creative_a = np.asarray(group['creativeID'].tolist())
            if creative_a[n - 1] != creative_a[n - 2]:
                clickTime_a = np.asarray(group['clickTime'].tolist())
                label_a = np.asarray(group['label'].tolist())
                values2write = [list(group.index)[n - 1], clickTime_a[n - 1], 1]
                to_file.write(','.join(str(v) for v in values2write) + '\n')
                if int(label_a[n - 1]) == 1:
                    positive_count += 1
                count += 1
                if count % 10000 == 0:
                    print count
    print positive_count / float(count)
    to_file.close()


def split_train_test(dir_path):
    train_test_fea = pd.read_csv(dir_path+'train_test_last_click_fea.csv')
    print train_test_fea.head()
    train_fea = train_test_fea[train_test_fea['clickTime'] < 310000]
    test_fea = train_test_fea[train_test_fea['clickTime'] >= 310000]
    train_fea.sort('instance', inplace=True)
    test_fea.sort('instance', inplace=True)
    train_fea.to_csv(dir_path + 'train_time_last_click.csv', index=False)
    test_fea.to_csv(dir_path + 'predict_time_last_click.csv', index=False)


# 读取最后点击特征
def load_user_last_click(dir_path):
    header = ['instance', 'clickTime', 'different_last_creative']

    train_df = pd.read_csv(dir_path + 'train_time_last_click.csv')
    train_values = np.asarray(train_df.values).astype(int)
    train_dict = {}
    for row in train_values:
        train_dict[int(row[0])] = int(row[2])

    predict_df = pd.read_csv(dir_path + 'predict_time_last_click.csv')
    predict_values = np.asarray(predict_df.values).astype(int)
    predict_dict = {}
    for row in predict_values:
        predict_dict[int(row[0])] = int(row[2])
    del train_df, train_values, predict_df, predict_values
    print "Loading last click finished."
    return train_dict, predict_dict



if __name__ == '__main__':
    split_train_test(constants.project_path+'/dataset/custom/')
