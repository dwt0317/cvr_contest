# -*- coding:utf-8 -*-
from util import constants
import numpy as np
import pandas as pd

from util.utils import list2dict


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
    train_f = open(constants.cus_train_path)
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
        tmp = [int(fields[0])] + ad_dict[int(fields[3])]
        ad_train.append(tmp)

    a = np.array(ad_train, dtype=int)
    print a[:5]
    print a.shape
    file_header = "label,creativeID,adID,campaignID,advertiserID,appID,appPlatform"
    with open(to_path, 'w') as f:
        np.savetxt(f, a, delimiter=',', fmt="%s", header=file_header)


# 生成广告特征 return {creativeID:(adID,campaignID,advertiserID), appID,appPlatform,appCategory}
def build_ad_feature(has_sparse=True):
    train_df = pd.read_csv(constants.project_path + "/dataset/raw/" + "ad.csv")

    ad_id_lens = [0, 0, 0, 0, 0]
    appID_set = list2dict(train_df['appID'].unique())
    campID_set = list2dict(train_df['campaignID'].unique())
    adverID_set = list2dict(train_df['advertiserID'].unique())
    ad_id_lens[2] = len(campID_set) + 1
    ad_id_lens[3] = len(adverID_set) + 1
    ad_id_lens[4] = len(appID_set) + 1
    if has_sparse:
        # 将list转为dict{id:index}
        creativeID_set = list2dict(train_df['creativeID'].unique())
        adID_set = list2dict(train_df['adID'].unique())
        ad_id_lens = [len(creativeID_set) + 1, len(adID_set) + 1, len(campID_set) + 1, len(adverID_set) + 1,
                      len(appID_set) + 1]
    print "Building ID set finished."

    # read APP category
    category_dict = {}
    with open(constants.project_path + "/dataset/raw/" + "app_categories.csv") as cate_f:
        cate_f.readline()
        for line in cate_f:
            fields = line.strip().split(',')
            category_dict[int(fields[0])] = int(fields[1])
    # category = list2dict(list(set(category_dict.values())))
    print "Building category set finished."

    # read feature file
    f = open(constants.project_path + "/dataset/raw/" + "ad.csv")
    ad_feature = {}
    offset = 0
    f.readline()
    for line in f:
        line_feature = []
        fields = line.strip().split(',')
        offset = 0
        if has_sparse:
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

        # 下面的id特征非常稠密，不与一般id类特征一同处理
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

        if int(fields[4]) in appID_set:
            line_feature.append(offset+appID_set[int(fields[4])])
        else:
            line_feature.append(offset)
        offset += ad_id_lens[4]
        # appPlatform
        line_feature.append(offset+int(fields[5]))
        offset += 3

        # app category one hot, category分为一级和二级类目
        app_cate = int(category_dict[int(fields[4])])
        if app_cate >= 100:
            app_cate_1 = app_cate/100
            app_cate_2 = app_cate
        else:
            app_cate_1 = app_cate
            app_cate_2 = app_cate_1 * 100
        line_feature.append(offset+app_cate_1)
        offset += 10

        if has_sparse:
            line_feature.append(offset+app_cate_2)
            offset += 1000

        ad_feature[int(fields[0])] = line_feature

    f.close()
    print "Building ad feature finished."
    return ad_feature, offset


if __name__ == '__main__':
    # build_ad_train(constants.project_path + "/dataset/custom/train_ad_label.csv")
    pos, lent = build_ad_feature(has_sparse=True)
    print lent
    for key in pos.keys()[:10]:
        print pos[key]
