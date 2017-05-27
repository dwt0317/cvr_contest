# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

from util import *
import math


# 30天之前的用户app安装特征
def user_before_app_feature():
    installedapp_file = constants.project_path+"/dataset/raw/user_installedapps.csv"
    user_app_df = pd.read_csv(installedapp_file)
    user_counts = user_app_df['userID'].value_counts()
    max_app = user_counts.max()
    min_app = user_counts.min()
    median_app = user_counts.median()
    user_app_dict = user_counts.to_dict()
    print "Building user app dict finished."
    return user_app_dict, max_app, min_app, median_app


# 最近30天用户APP安装特征
def user_recent_app_feature():
    user_action_file = constants.project_path+"/dataset/raw/user_installedapps.csv"
    user_app_df = pd.read_csv(user_action_file)
    user_counts = user_app_df['userID'].value_counts()
    max_app = user_counts.max()
    min_app = user_counts.min()
    median_app = user_counts.median()
    user_app_dict = user_counts.to_dict()
    print "Building user app dict finished."
    return user_app_dict, max_app, min_app, median_app


# user最常装的app类别
def user_recent_app_category():
    # 将用户favorite category写入文件
    user_category_file = constants.project_path + "/dataset/raw/" + "user_app_actions_with_category.csv"
    # with open(user_category_file, '')

    user_category_df = pd.read_csv(user_category_file)
    # favorite = user_category_df.groupby(['userID']).agg(lambda x: x['appCategory'].value_counts().index[0])
    diversity = user_category_df.groupby(['userID'])['appCategory'].unique()
    print diversity.groupby(['appCategory'])

    # num = user_category_df.groupby(['userID']).agg(lambda x: len(x['appCategory'].values))
    # f_d = pd.merge(left=favorite, right=diversity, on='userID')
    # del favorite, diversity
    # f_d_n = pd.merge(left=f_d, right=num, on='userID')
    # del num
    # print f_d_n.head(10)
    # favorite.to_csv(constants.project_path + "/dataset/raw/" + "user_app_favorite_more.csv", columns=['appCategory'])


    # user_category_file = constants.project_path + "/dataset/custom/" + "user_app_favorite.csv"
    # user_favorite_dict = pd.read_csv(user_category_file).to_dict()
    # return user_favorite_dict


# age * 9, gender * 3, education * 8, marriage * 4, baby * 7, residence * 35,
def build_user_profile(has_id=False):
    # make user raw data
    f = open(constants.project_path + "/dataset/raw/" + "user.csv")
    # user_app_dict, max_app, min_app, median_app = user_before_app_feature()

    # user id 特征
    stat = pd.read_csv(constants.clean_train_path)
    userdf = stat['userID'].value_counts()
    userlist = []
    for i, row in userdf.iteritems():
        if int(row) > 2:
            userlist.append(i)
    userIDset = utils.list2dict(userlist)

    user_favorite_dict = user_recent_app_category()
    # user编号从1开始的, 0处补一个作为填充
    user_feature = [[0]]
    offset = 0
    f.readline()
    for line in f:
        offset = 0
        features = []
        fields = line.strip().split(',')

        # age
        age = int(fields[1])
        if age == 0:
            features.append(age)
        else:
            features.append(((age-1) % 10)+1)
        offset += 9
        # gender
        features.append(offset+int(fields[2]))
        offset += 3
        # education
        features.append(offset+int(fields[3]))
        offset += 8
        # marriage
        features.append(offset+int(fields[4]))
        offset += 4
        # baby
        features.append(offset+int(fields[5]))
        offset += 7
        # residence
        features.append(offset+int(fields[6])/100)
        offset += 35

        # favorite category
        userID = int(fields[0])
        if userID in user_favorite_dict:
            features.append(offset+int(user_favorite_dict[userID]))
        else:
            features.append(offset)
        offset += 10

        if has_id:
            if userID in userIDset:
                features.append(offset+userIDset[userID])
            else:
                features.append(0)
            offset += len(userIDset) + 1

        # # installed app
        # user_id = int(fields[0])
        # if user_id in user_app_dict:
        #     # features.append(round((user_app_dict[int(fields[0])] - min_app) / (max_app - min_app)), 5)
        #     features.append(user_app_dict[int(fields[0])])
        # else:
        #     # features.append(round((median_app - min_app) / (max_app - min_app)), 5)
        #     features.append(0)
        # offset += 1
        #
        user_feature.append(features)

    print "Buliding user profile finished."
    return user_feature, offset


def count_user_freq():
    train_df = pd.read_csv(constants.train_path)
    querylist = []
    querydf = train_df['userID'].value_counts()
    for i, row in querydf.iteritems():
        querylist.append(int(row))
    a = np.array(querylist)
    print "user 20%: " + str(np.percentile(a, 20))
    print "user 40%: " + str(np.percentile(a, 40))
    print "user 60%: " + str(np.percentile(a, 60))
    print "user 80%: " + str(np.percentile(a, 80))
    print "user 90%: " + str(np.percentile(a, 90))
    print "user 95%: " + str(np.percentile(a, 95))


if __name__ == '__main__':
    user_recent_app_category()
    # count_user()
    # u, m, n = user_before_app_feature()
    # for k in u.keys()[:30]:
    #     print k, u[k]