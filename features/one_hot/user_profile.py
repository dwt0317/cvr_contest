# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

from util import constants


def user_app_feature():
    installedapp_file = constants.project_path+"/dataset/raw/user_installedapps.csv"
    user_app_df = pd.read_csv(installedapp_file)
    user_counts = user_app_df['userID'].value_counts()
    max_app = user_counts.max()
    min_app = user_counts.min()
    median_app = user_counts.median()
    user_app_dict = user_counts.to_dict()
    print "Building user app dict finished."
    return user_app_dict, max_app, min_app, median_app


# age * 9, gender * 3, education * 8, marriage * 4, baby * 7, residence * 35,
def build_user_profile():
    # make user raw data
    f = open(constants.project_path + "/dataset/raw/" + "user.csv")
    user_app_dict, max_app, min_app, median_app = user_app_feature()
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

        # installed app
        user_id = int(fields[0])
        if user_id in user_app_dict:
            features[offset] = (user_app_dict[int(fields[0])] - min_app) / (max_app - min_app)
        else:
            features[offset] = (median_app - min_app) / (max_app - min_app)
        offset += 1

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
    u, m, n = user_app_feature()
    for k in u.keys()[:30]:
        print k, u[k]