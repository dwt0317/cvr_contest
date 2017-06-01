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
    adverID_set = list2dict(train_df['advertiserID'].unique())
    campID_set = list2dict(train_df['campaignID'].unique())
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
    del train_df
    # read APP category
    category_dict = {}
    with open(constants.project_path + "/dataset/raw/" + "app_categories.csv") as cate_f:
        cate_f.readline()
        for line in cate_f:
            fields = line.strip().split(',')
            category_dict[int(fields[0])] = int(fields[1])

    print "Building category set finished."
    ad_map = {}
    # read feature file
    f = open(constants.project_path + "/dataset/raw/" + "ad.csv")
    ad_feature = {}
    offset = 0
    f.readline()
    for line in f:
        line_feature = []
        fields = line.strip().split(',')
        line_map = fields[1:]
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

        if int(fields[2]) in campID_set:
            line_feature.append(offset+campID_set[int(fields[2])])
        else:
            line_feature.append(offset)
        offset += ad_id_lens[2]

        # 下面的id特征非常稠密，不与一般id类特征一同处理
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
        line_map.append(app_cate_1)
        if has_sparse:
            line_feature.append(offset+app_cate_2)
            offset += 1000

        ad_feature[int(fields[0])] = line_feature
        ad_map[int(fields[0])] = line_map

    f.close()
    print "Building ad feature finished."
    return ad_feature, ad_map, offset


def build_ad_spec_feature(has_sparse=True):
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
    print "Building category set finished."
    # read feature file
    f = open(constants.project_path + "/dataset/raw/" + "ad.csv")
    ad_feature = {}
    offset = 0
    f.readline()
    appID_top = list2dict([442, 262, 109, 472, 420, 218, 271, 83, 419, 88, 428, 286, 319, 421,
                           284, 206, 146, 383, 105, 360])
    adver_top = list2dict([90, 11, 81, 4, 22, 89, 75, 28, 31, 32, 57, 21, 69, 85, 29, 53, 86, 56, 33, 39])
    for line in f:
        line_feature = []
        fields = line.strip().split(',')
        offset = 0
        if has_sparse:
            if int(fields[0]) in creativeID_set:
                line_feature.append(offset + creativeID_set[int(fields[0])])
            else:
                line_feature.append(offset)
            offset += ad_id_lens[0]

            if int(fields[1]) in adID_set:
                line_feature.append(offset + adID_set[int(fields[1])])
            else:
                line_feature.append(offset)
            offset += ad_id_lens[1]

        if int(fields[2]) in campID_set:
            line_feature.append(offset + campID_set[int(fields[2])])
        else:
            line_feature.append(offset)
        offset += ad_id_lens[2]

        # 下面的id特征非常稠密，不与一般id类特征一同处理
        if int(fields[3]) in adver_top:
            line_feature.append(offset + adverID_set[int(fields[3])])
        else:
            line_feature.append(offset)
        offset += len(adver_top) + 1

        if int(fields[4]) in appID_top:
            line_feature.append(offset + appID_top[int(fields[4])])
        else:
            line_feature.append(offset)

        offset += len(appID_top) + 1

        # appPlatform
        line_feature.append(offset + int(fields[5]))
        offset += 3

        # app category one hot, category分为一级和二级类目
        app_cate = int(category_dict[int(fields[4])])
        if app_cate >= 100:
            app_cate_1 = app_cate / 100
        else:
            app_cate_1 = app_cate
        line_feature.append(offset + app_cate_1)
        offset += 10

        ad_feature[int(fields[0])] = line_feature

    f.close()
    print "Building ad feature finished."
    return ad_feature, offset


if __name__ == '__main__':
    pass
    # build_ad_train(constants.project_path + "/dataset/custom/train_ad_label.csv")
    # pos, lent = build_ad_feature(has_sparse=True)
    # print lent
    # for key in pos.keys()[:10]:
    #     print pos[key]
