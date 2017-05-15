# -*- coding:utf-8 -*-
import constants
import pandas as pd
import numpy as np
from Utils import list2dict


def ad():
    train_df = pd.read_csv(constants.project_path + "/dataset/custom/train_c_time.csv")
    positive = train_df[train_df['label'] == 1]

    head = 'telecomsOperator'

    values = train_df[head].value_counts()
    p_values = positive[head].value_counts()
    # print values
    # print p_values
    # print p_values.iloc[0]/float(values.iloc[0])
    # print p_values.iloc[0]/float(values.iloc[1])

    log_file = open(constants.project_path + "/statistic/statistic", "a")
    log_file.write("head: " + head + '\n')
    for i in xrange(len(values)):
        if len(p_values) > i:
            log_file.write(str(i) + ": " + str(p_values.loc[i]/float(values.loc[i])) + '\n')
    log_file.write('\n' + '\n')
    log_file.close()


def user():
    train_df = pd.read_csv(constants.project_path + "/dataset/custom/train_c_time.csv")
    user_df = pd.read_csv(constants.project_path + "/dataset/raw/" + "user.csv")
    merge_df = pd.merge(left=train_df, right=user_df, on=['userID'])
    print merge_df.head(5)

    positive = merge_df[merge_df['label'] == 1]
    head = 'education'
    values = merge_df[head].value_counts()
    p_values = positive[head].value_counts()

    log_file = open(constants.project_path + "/statistic/statistic", "a")
    log_file.write("head: " + head + '\n')
    for i in xrange(len(values)):
        if len(p_values) > i:
            log_file.write(str(i) + ": " + str(p_values.loc[i]/float(values.loc[i])) + '\n')
    log_file.write('\n' + '\n')
    log_file.close()


def position():
    train_df = pd.read_csv(constants.project_path + "/dataset/custom/train_c_time.csv")
    pos_df = pd.read_csv(constants.project_path + "/dataset/raw/" + "position.csv")
    merge_df = pd.merge(left=train_df, right=pos_df, on=['positionID'])
    print merge_df.head(5)

    positive = merge_df[merge_df['label'] == 1]
    head = 'positionType'
    values = merge_df[head].value_counts()
    p_values = positive[head].value_counts()

    log_file = open(constants.project_path + "/statistic/statistic", "a")
    log_file.write("head: " + head + '\n')
    for i in xrange(len(values)):
        if len(p_values) > i:
            log_file.write(str(i) + ": " + str(p_values.loc[i]/float(values.loc[i])) + '\n')
    log_file.write('\n' + '\n')
    log_file.close()


if __name__ == '__main__':
    position()
