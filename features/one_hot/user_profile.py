# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

from util import constants


def user_app_feature():
    user_app_dict = [0]
    with open(constants.project_path+"/dataset/raw/user_installedapps.csv") as f:
        pre_ID = -1
        f.readline()
        for line in f:
            row = line.strip().split(',')
            userID = int(row[0])
            appID = int(row[1])
            if userID == pre_ID:
                user_app_dict[userID] += 1
            else:
                user_app_dict.append(1)
            pre_ID = userID
    df = pd.DataFrame(user_app_dict)
    print df.shape
    return user_app_dict


# age * 9, gender * 3, education * 8, marriage * 4, baby * 7, residence * 35,
def build_user_profile():
    # make user raw data
    f = open(constants.project_path + "/dataset/raw/" + "user.csv")
    # user编号从1开始的
    user_feature = [{0: 0}]
    offset = 0
    f.readline()
    for line in f:
        offset = 0
        features = {}
        fields = line.strip().split(',')

        age = int(fields[1])
        if age == 0:
            features[age] = 1
        else:
            features[((age-1) % 10)+1] = 1
        offset += 9
        # gender
        features[offset+int(fields[2])] = 1
        offset += 3
        # education
        features[offset+int(fields[3])] = 1
        offset += 8
        # marriage
        features[offset+int(fields[4])] = 1
        offset += 4
        # baby
        features[offset+int(fields[5])] = 1
        offset += 7
        # residence
        features[offset+int(fields[6])/100] = 1
        offset += 35
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
    # count_user()
    user_app_feature()