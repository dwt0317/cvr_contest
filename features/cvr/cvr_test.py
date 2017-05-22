# -*- coding:utf-8 -*-
import copy
import pandas as pd
from util import constants

'''
用于生成线上test样本的统计类特征
'''

alpha = 135  # for smoothing
beta = 5085

# 处理cvr的统一方法
def cvr_helper(total_df, header, dim, feature_map):
    '''
        total_df: dataframe
        header: 要处理的维度
        dim: 该维度的维数
        feature_map: 记录map
    '''
    # [,..,] * dim
    feature = []
    max_click = 0
    min_click = len(total_df) + 1
    max_cv = 0
    min_cv = len(total_df) + 1
    for category in xrange(dim):
        feature_list = []
        this_df = total_df[total_df[header] == category]
        click_num = len(this_df)
        conversion_num = len(total_df[(total_df.label > 0) & (total_df[header] == category)])
        conversion_rate = round(float(conversion_num + alpha * beta) / float(click_num + beta), 5)
        max_click = max(max_click, click_num)
        min_click = min(min_click, click_num)
        max_cv = max(max_cv, conversion_num)
        min_cv = min(min_cv, conversion_num)
        # feature_list.append(int(click_num))
        # feature_list.append(int(conversion_num))
        feature_list.append(conversion_rate)
        feature.append(feature_list)
    # 归一化
    # for i in xrange(dim):
    #     feature[i][0] = round((feature[i][0] - min_click) / float(max_click - min_click), 5)
    #     feature[i][1] = round((feature[i][1] - min_cv) / float(max_cv - min_cv), 5)
    feature_map[header] = feature


'''
处理user相关cvr数据
'''


# 统计用户画像维度的点击率
def user_profile_cvr(file_path):
    total_df = pd.read_csv(file_path)
    # {header:[[3],[3],...,[3]], }
    user_features = {}
    cvr_helper(total_df, 'gender', 3, user_features)
    cvr_helper(total_df, 'education', 9, user_features)
    cvr_helper(total_df, 'marriageStatus', 5, user_features)
    cvr_helper(total_df, 'haveBaby', 7, user_features)

    print "Building user profile cvr finished."
    del total_df
    return user_features


# 统计userID维度的转化率
def userID_cvr():
    alpha = 0.0248  # for smoothing
    beta = 75

    user_stat = pd.read_csv(constants.project_path + "/dataset/raw/user.csv")
    stat_file = open(constants.cus_train_path, 'r')
    userID_set = user_stat['userID'].values
    del user_stat
    click_set = {}
    cv_set = {}
    # skip header
    stat_file.readline()
    # 采用按行读取的方式，统计各个id的点击和转化
    for line in stat_file:
        row = line.strip().split(',')
        click_set[row[4]] = 1 + click_set.setdefault(row[4], 0)
        if int(row[0]) == 1:
            cv_set[row[4]] = 1 + cv_set.setdefault(row[4], 0)
    stat_file.close()

    userID_cvr = {}
    # 根据id计算转化率
    for id in userID_set:
        click = float(click_set.setdefault(id, 0))
        cv = float(cv_set.setdefault(id, 0))
        cvr = (cv + alpha * beta) / (click + beta)
        userID_cvr[id] = round(cvr, 5)
    print "Building userID cvr finished."
    return userID_cvr


# {userID1:[cvr1, cvr2,..], userID2:[cvr1, cvr2,..], ...}
def build_user_cvr(train_dir):
    userID_feature = userID_cvr()
    user_pro_feature = user_profile_cvr(train_dir+"train_with_user_info.csv")
    user_cvr_features = {}
    user_file = open(constants.project_path + "/dataset/raw/user.csv", 'r')
    user_file.readline()
    for line in user_file:
        row = line.strip().split(',')
        feature_list = [userID_feature[int(row[0])]]
        # feature_list = copy.copy(user_pro_feature['gender'][int(row[2])])
        feature_list.extend(user_pro_feature['gender'][int(row[2])])
        feature_list.extend(user_pro_feature['education'][int(row[3])])
        feature_list.extend(user_pro_feature['marriageStatus'][int(row[4])])
        feature_list.extend(user_pro_feature['haveBaby'][int(row[5])])
        user_cvr_features[int(row[0])] = feature_list
    print "Building user cvr feature finished."
    return user_cvr_features


'''
处理position相关cvr数据
'''


# 统计position 各维度cvr数据
def pos_info_cvr(pos_info_file):
    total_df = pd.read_csv(pos_info_file)
    # {header:[[3],[3],...,[3]], }
    pos_features = {}
    cvr_helper(total_df, 'sitesetID', 3, pos_features)
    cvr_helper(total_df, 'positionType', 6, pos_features)
    return pos_features


# 按positionID处理cvr数据
def build_pos_cvr(train_dir):
    pos_info_feature = pos_info_cvr(train_dir+"train_with_pos_info.csv")
    pos_file = open(constants.project_path + "/dataset/raw/position.csv", 'r')
    pos_file.readline()
    pos_cvr_features = {}
    i = 1
    for line in pos_file:
        row = line.strip().split(',')
        feature_list = copy.copy(pos_info_feature['sitesetID'][int(row[1])])
        feature_list.extend(pos_info_feature['positionType'][int(row[2])])
        pos_cvr_features[int(row[0])] = feature_list
        # print pos_cvr_features[int(row[0])]
    return pos_cvr_features


'''
处理广告相关cvr数据
'''


def getADFeature(filePath):
    '''
    :type filePath: string train_with_ad.csv的路径
    :rtype: dict{creativeID:[ad_cl,ad_cv,ad_cvr,
                            campaign_cl,campaign_cv,campaign_cvr
                            advertiser_cl,advertisr_cv,advertiser_cvr
                            app_cl,app_cv,app_cvr
                            appPlatform_cl,appPlatform_cv,appPlatform_cvr]}
    :Example: a = getADFeature('train_with_ad.csv')
    '''
    # ad_data = pd.read_csv(filePath)
    ad_file = open(filePath, 'r')
    # length = len(ad_data)
    ad = {}
    campaign = {}
    advertiser = {}
    app = {}
    appPlatform = {}
    creativeID_adFeature_map = {}

    for line in ad_file:
        row = line.strip().split(',')
        adID_key = row[8]
        campaignID_key = row[9]
        advertiserID_key = row[10]
        appID_key = row[11]
        appPlatform_key = row[12]

        # 更新adID的数据
        if adID_key not in ad:
            ad[adID_key] = [0, 0, 0]
        if row[1] == '1':
            ad[adID_key][0] += 1
            ad[adID_key][1] += 1
        else:
            ad[adID_key][0] += 1
        if ad[adID_key][0] != 0:
            ad[adID_key][2] = round((float(ad[adID_key][1])+alpha * beta) / (float(ad[adID_key][0]) + beta), 5)

        # 更新campaignID的数据
        if campaignID_key not in campaign:
            campaign[campaignID_key] = [0, 0, 0]
        if row[1] == '1':
            campaign[campaignID_key][0] += 1
            campaign[campaignID_key][1] += 1
        else:
            campaign[campaignID_key][0] += 1
        if campaign[campaignID_key][0] != 0:
            campaign[campaignID_key][2] = round((float(campaign[campaignID_key][1]) + alpha * beta) /
                                                (float(campaign[campaignID_key][0]) + beta), 5)

        # 更新advertiserID的数据
        if advertiserID_key not in advertiser:
            advertiser[advertiserID_key] = [0, 0, 0]
        if row[1] == '1':
            advertiser[advertiserID_key][0] += 1
            advertiser[advertiserID_key][1] += 1
        else:
            advertiser[advertiserID_key][0] += 1
        if advertiser[advertiserID_key][0] != 0:
            advertiser[advertiserID_key][2] = round((float(advertiser[advertiserID_key][1]) + alpha * beta) / (float(
                advertiser[advertiserID_key][0]) + beta), 5)

        # 更新appID的数据
        if appID_key not in app:
            app[appID_key] = [0, 0, 0]
        if row[1] == '1':
            app[appID_key][0] += 1
            app[appID_key][1] += 1
        else:
            app[appID_key][0] += 1
        if app[appID_key][0] != 0:
            app[appID_key][2] = round((float(app[appID_key][1]) + alpha * beta) / (float(app[appID_key][0]) + beta), 5)

        # 更新appID的数据
        if appPlatform_key not in appPlatform:
            appPlatform[appPlatform_key] = [0, 0, 0]
        if row[1] == '1':
            appPlatform[appPlatform_key][0] += 1
            appPlatform[appPlatform_key][1] += 1
        else:
            appPlatform[appPlatform_key][0] += 1
        if appPlatform[appPlatform_key][0] != 0:
            appPlatform[appPlatform_key][2] = round((float(appPlatform[appPlatform_key][1]) + alpha * beta) / (float(
                appPlatform[appPlatform_key][0]) + beta), 5)

    # 获取最终的list
    ad_file = open(filePath, 'r')
    for line in ad_file:
        row = line.strip().split(',')
        adID_data = ad[row[8]]
        campaignID_data = campaign[row[9]]
        advertiserID_data = advertiser[row[10]]
        appID_data = app[row[11]]
        appPlatform_data = appPlatform[row[12]]

        creativeData = adID_data + campaignID_data + advertiserID_data + appID_data + appPlatform_data
        creativeID_adFeature_map[row[3]] = creativeData

    return creativeID_adFeature_map


if __name__ == "__main__":
    user_cvr_feature = build_pos_cvr()
    for id in user_cvr_feature.keys()[:5]:
        print user_cvr_feature[id]
