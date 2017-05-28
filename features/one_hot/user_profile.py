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


# user最常装的app类别
def user_recent_app_category():
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


# 30天前用户app类别特征
def user_before_app_category():
    user_before_category_file = constants.project_path + \
                                "/dataset/custom/favorite/user_installedapps_with_category_group.csv"
    # to_file = open(constants.project_path+"/dataset/custom/favorite/user_before_app_favorite.csv", 'w')
    # to_file.write('userID,appCategory,diversity\n')
    user_before_favorite_dict = {}
    with open(user_before_category_file, 'r') as f:
        f.readline()
        preID= -1
        diversity = 0
        # count, category
        max_pair = [0, 0]
        for line in f:
            row = line.strip().split(',')
            userID = int(row[0])
            if userID == preID:
                if int(row[2]) > max_pair[0]:
                    max_pair = [int(row[2]), int(row[1])]
            else:
                user_before_favorite_dict[preID] = [max_pair[1], diversity]
                # to_file.write(str(preID)+','+str(max_pair[1])+','+str(diversity)+'\n')
                max_pair = [0, 0]
                diversity = 0
            preID = userID
    #         diversity += 1

    print "Building user before app favorite finished."
    return user_before_favorite_dict


# age * 9, gender * 3, education * 8, marriage * 4, baby * 7, residence * 35,
def build_user_profile(has_sparse=False):
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
    user_before_favorite_dict = user_before_app_category()
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
        # hometown
        features.append(offset+int(fields[6])/100)
        offset += 35
        # if has_sparse:
        #     # detail hometown
        #     features.append(offset + int(fields[6]))
        #     offset += 3500
        # residence
        # features.append(offset + int(fields[7]) / 100)
        # offset += 35
        # if has_sparse:
        #     # detail residence
        #     features.append(offset + int(fields[7]))
        #     offset += 3500

        # favorite category remove to test gbdt
        userID = int(fields[0])
        # if userID in user_favorite_dict:
        #     features.append(offset+int(user_favorite_dict[userID][0]))
        #     offset += 10
        #     features.append(offset+int(user_favorite_dict[userID][1]))
        # else:
        #     features.append(offset)
        #     offset += 10
        #     features.append(offset)
        # offset += 10

        # before favorite category
        if userID in user_before_favorite_dict:
            features.append(offset+int(user_before_favorite_dict[userID][0]))
            # offset += 10
            # features.append(offset+int(user_before_favorite_dict[userID][1]))
        else:
            features.append(offset)
            # offset += 10
            # features.append(offset)
        offset += 10


        if has_sparse:
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
    # user_before_app_category()
    # count_user()
    u = user_before_app_category()
    # for k in u.keys()[:30]:
    #     print k, u[k]