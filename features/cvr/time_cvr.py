# -*- coding:utf-8 -*-
import copy
import pandas as pd
from util import constants
import math

'''
用于生成时序特征
'''

# 用户安装行为
def build_user_action():
    # 读取favorite文件
    user_category_file = constants.project_path + "/dataset/custom/" + "user_app_actions_with_category.csv"
    user_action_dict = {}
    with open(user_category_file, 'r') as f:
        f.readline()
        for line in f:
            row = line.strip().split(',')
            userID = int(row[0])
            day = int(row[1]) / 10000
            category = int(row[3])
            if day in user_action_dict:
                if userID in user_action_dict[day]:
                    user_action_dict[day][userID][category] += 1
                else:
                    user_action_dict[day][userID] = {category:1}
            else:
                user_action_dict[day] = {userID: {category: 1}}
    return user_action_dict


def user_backup():
    # 将用户favorite category写入文件
    # user_category_file = constants.project_path + "/dataset/custom/" + "user_app_actions_with_category.csv"
    # user_category_df = pd.read_csv(user_category_file)
    # favorite = pd.read_csv(constants.project_path + "/dataset/custom/" + "user_app_favorite.csv")
    # diversity = user_category_df.groupby(['userID']).agg(lambda x: len(x['appCategory'].unique()))
    # df_fi = diversity.to_frame()
    # df_fi['userID'] = diversity.index
    # f_d = pd.merge(favorite, df_fi, on=['userID'])
    # f_d.columns = ['userID', 'appCategory', 'diversity']
    # f_d.astype(int).to_csv(constants.project_path + "/dataset/custom/favorite/" + "user_app_favorite_more.csv",
    #                        index=False)

    # 读取favorite文件
    user_category_file = constants.project_path + "/dataset/custom/favorite/" + "user_app_favorite_more.csv"
    user_favorite_dict = {}
    with open(user_category_file, 'r') as f:
        f.readline()
        for line in f:
            row = line.strip().split(',')
            l = [int(row[1]), int(row[2])]
            user_favorite_dict[int(row[0])] = l
    return user_favorite_dict


if __name__ == '__main__':
    pass