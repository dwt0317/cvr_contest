# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

from util import *
import copy


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


# age * 11, gender * 3, education * 8, marriage * 4, baby * 7, residence * 35,
def build_user_profile(has_sparse=False):
    # make user raw data
    f = open(constants.project_path + "/dataset/raw/" + "user.csv")
    # user_app_dict, max_app, min_app, median_app = user_before_app_feature()

    if has_sparse:
        # user id 特征
        stat = pd.read_csv(constants.clean_train_path)
        userdf = stat['userID'].value_counts()
        del stat
        userlist = []
        for i, row in userdf.iteritems():
            if int(row) > 2:
                userlist.append(i)
        userIDset = utils.list2dict(userlist)
        del userlist

    # user编号从1开始的, 0处补一个作为填充
    user_feature = [[0]]
    user_map = [[0]]
    offset = 0
    f.readline()
    for line in f:
        offset = 0
        features = []
        fields = line.strip().split(',')
        or_features = []

        # age
        age_bucket = [7, 12, 16, 21, 25, 32, 40, 48, 55, 65]
        age = int(fields[1])
        if age != 0:
            for i in xrange(len(age_bucket)):
                if age < age_bucket[i]:
                    age = i+1
                    break
            if age > age_bucket[9]:
                age = 11
        features.append(age)
        or_features.append(age)
        offset += 12

        # gender
        features.append(offset+int(fields[2]))
        or_features.append(int(fields[2]))
        offset += 3
        # education
        features.append(offset+int(fields[3]))
        or_features.append(int(fields[3]))
        offset += 8
        # marriage
        features.append(offset+int(fields[4]))
        or_features.append(int(fields[4]))
        offset += 4
        # baby
        features.append(offset+int(fields[5]))
        or_features.append(int(fields[5]))
        offset += 7
        # hometown
        features.append(offset+int(fields[6])/100)
        offset += 35
        # if has_sparse:
        #     # detail hometown
        #     features.append(offset + int(fields[6]))
        #     offset += 3500
        # residence
        features.append(offset + int(fields[7]) / 100)
        offset += 35
        # if has_sparse:
        #     # detail residence
        #     features.append(offset + int(fields[7]))
        #     offset += 3500

        # favorite category remove to test gbdt
        userID = int(fields[0])
        or_features.append(int(fields[6])/100)
        or_features.append(int(fields[7])/100)

        # before favorite category
        features = [0, 0, 0, 0, 0]      # diversity, num, most, medium, low

        if has_sparse:
            if userID in userIDset:
                features.append(offset+userIDset[userID])
            else:
                features.append(0)
            offset += len(userIDset) + 1

        user_feature.append(features)
        user_map.append(or_features)
        # del userIDset
    print "Buliding user profile finished."
    return user_feature, user_map, offset


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
    build_user_profile()