# -*- coding:utf-8 -*-
import constants
import pandas as pd
import numpy as np
from Utils import list2dict

# 统计id类值出现频率
def count_id_freq():
    train_df = pd.read_csv(constants.project_path + "/dataset/custom/" + "train_ad_statistic.csv")
    adlist = []
    addf = train_df['adID'].value_counts()
    for i, row in addf.iteritems():
        adlist.append(int(row))
    a = np.array(adlist)
    print a
    print "ad 20%: " + str(np.percentile(a, 20))

    querylist = []
    querydf = train_df['camgaignID'].value_counts()
    for i, row in querydf.iteritems():
        querylist.append(int(row))
    a = np.array(querylist)
    print "camgaignID 20%: " + str(np.percentile(a, 20))

    querylist = []
    querydf = train_df['advertiserID'].value_counts()
    for i, row in querydf.iteritems():
        querylist.append(int(row))
    a = np.array(querylist)
    print "advertiserID 20%: " + str(np.percentile(a, 20))

    querylist = []
    querydf = train_df['appID'].value_counts()
    for i, row in querydf.iteritems():
        querylist.append(int(row))
    a = np.array(querylist)
    print "appID 20%: " + str(np.percentile(a, 20))


# 生成用于统计ad ID类信息的训练集文件
def build_ad_train(to_path):
    train_f = open(constants.train_path)
    train_f.readline()
    ad_f = open(constants.project_path + "/dataset/raw/" + "ad.csv")
    ad_f.readline()
    ad_dict = {}
    for line in ad_f:
        fields = line.strip().split(',')
        ad_dict[int(fields[0])] = fields

    ad_train = []
    for line in train_f:
        fields = line.strip().split(',')
        ad_train.append(ad_dict[int(fields[3])])

    a = np.array(ad_train, dtype=int)
    print a[:5]
    print a.shape
    file_header = "creativeID,adID,campaignID,advertiserID,appID,appPlatform"
    with open(to_path, 'w') as f:
        np.savetxt(f, a, delimiter=',', fmt="%s", header=file_header)


# 生成广告特征 return {creativeID:(adID,campaignID,advertiserID), appID,appPlatform}
def build_ad_id(has_id=True):
    train_df = pd.read_csv(constants.project_path + "/dataset/raw/" + "ad.csv")

    if has_id:
        # 将list转为dict{id:index}
        creativeID_set = list2dict(train_df['creativeID'].unique())
        adID_set = list2dict(train_df['adID'].unique())
        campID_set = list2dict(train_df['campaignID'].unique())
        adverID_set = list2dict(train_df['advertiserID'].unique())
        appID_set = list2dict(train_df['appID'].unique())

        ad_id_lens = [len(creativeID_set)+1, len(adID_set)+1, len(campID_set)+1, len(adverID_set)+1, len(appID_set)+1]
    f = open(constants.project_path + "/dataset/raw/" + "ad.csv")
    ad_feature = {}
    f.readline()
    for line in f:
        line_feature = []
        fields = line.strip().split(',')
        offset = 0
        if has_id:
            if int(fields[0]) in creativeID_set:
                line_feature.append(offset+creativeID_set[int(fields[0])])
            else:
                line_feature.append(offset)
            offset += ad_id_lens[0]

            if int(fields[1]) in adID_set:
                line_feature.append(offset+adID_set[int(fields[1])])
            else:
                line_feature.append(offset)
            offset += ad_id_lens[1]

            if int(fields[2]) in campID_set:
                line_feature.append(offset+campID_set[int(fields[2])])
            else:
                line_feature.append(offset)
            offset += ad_id_lens[2]

            if int(fields[3]) in adverID_set:
                line_feature.append(offset+adverID_set[int(fields[3])])
            else:
                line_feature.append(offset)
            offset += ad_id_lens[3]

        # appID非常稠密，不与一般id类特征一同处理
        if int(fields[4]) in appID_set:
            line_feature.append(offset+appID_set[int(fields[4])])
        else:
            line_feature.append(offset)
        offset += ad_id_lens[4]
        # appPlatform
        line_feature.append(offset+int(fields[5]))
        ad_feature[int(fields[0])] = line_feature

    print "Building ad id feature finished."
    return ad_feature


if __name__ == '__main__':
    # build_ad_train(constants.project_path + "/dataset/custom/train_ad.csv")
    pos = build_ad_id()
    for key in pos.keys()[:10]:
        print pos[key]
