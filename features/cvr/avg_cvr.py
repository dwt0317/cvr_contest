# -*- coding:utf-8 -*-
import copy
import pandas as pd
from util import constants
import math
import cPickle as pickle
import numpy as np

'''
用于生成平均cvr特征特征
'''

# alpha = 135  # for smoothing
# beta = 5085

# alpha = 123  # for smoothing
# beta = 5000

alpha = 256  # for smoothing
beta = 9179

# connectionType cvr
alphas = [9, 225, 22, 1]
betas = [2085, 7246, 2527, 90]

connection_bias = [0.00383, 0.02917, 0.00823, 0.00705, 0.00648]

not_smooth = False
in_memory = False

left_day = 17
right_day = 24

# left_day = 24
# right_day = 31


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
    max_cvr = 0.0
    min_cvr = 1.0
    for category in xrange(dim):
        feature_list = []
        this_df = total_df[(total_df[header] == category)]
        click = len(this_df)
        cv = len(this_df[(this_df.label == 1)])
        if click == 0 and not_smooth:
            conversion_rate = 0
        else:
            conversion_rate = round(float(cv + alpha) / float(click + alpha + beta), 5)
        # max_click = max(max_click, click)
        # min_click = min(min_click, click)
        max_cv = max(max_cv, cv)
        min_cv = min(min_cv, cv)
        max_cvr = max(conversion_rate, max_cvr)
        min_cvr = min(conversion_rate, min_cvr)
        if cv != 0:
            cv = round(math.log(cv, 2), 5)
        if click != 0:
            click = round(math.log10(click), 5)

        feature_list.append(click)
        feature_list.append(cv)
        feature_list.append(conversion_rate)

        feature.append(feature_list)
    # 归一化
    for i in xrange(dim):
        # feature[i][0] = round((feature[i][0] - min_click) / float(max_click - min_click), 5)
        feature[i][0] = round((feature[i][0] - min_cvr) / float(max_cvr - min_cvr), 5)
        # feature[i][1] = round((feature[i][1] - min_cv) / float(max_cv - min_cv), 5)
    feature_map[header] = feature


# 统计ID维度的转化率, 所有的都用pos来计算，raw_file不再使用
def id_cvr_helper(raw_file, stat_dir, header, id_index):
    # stat = pd.read_csv(raw_file)
    stat_file = open(stat_dir+'train_with_pos_info.csv', 'r')
    id_set = set(np.loadtxt(constants.custom_path+'/idset/'+header, dtype=int))
    # del stat
    click_set = {}
    cv_set = {}
    # skip header
    stat_file.readline()
    # 采用按行读取的方式，统计各个id的点击和转化
    for line in stat_file:
        row = line.strip().split(',')
        # day = int(row[1])
        # if day < left_day * 10000 or day > right_day * 10000:
        #     continue
        id = int(row[id_index])
        click_set[id] = 1 + click_set.setdefault(id, 0)
        if int(row[0]) == 1:
            cv_set[id] = 1 + cv_set.setdefault(id, 0)
    stat_file.close()

    id_cvr = {}
    # 根据id计算转化率
    for id in id_set:
        click = float(click_set.setdefault(id, 0))
        cv = float(cv_set.setdefault(id, 0))
        if click == 0 and not_smooth:
            cvr = 0
        else:
            cvr = (cv + alpha) / (click + alpha + beta)
        if cv != 0:
            cv = round(math.log(cv, 2), 5)
        if click != 0:
            click = round(math.log10(click), 5)
        if id == 0:
            id_cvr[0] = [0, 0, round(cvr, 5)]
        else:
            id_cvr[id] = [click, cv, round(cvr, 5)]
        # id_cvr[id] = [cv, round(cvr, 5)]
    print "Building " + header + " cvr finished."
    return id_cvr



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


# {userID1:[cvr1, cvr2,..], userID2:[cvr1, cvr2,..], ...}
def build_user_cvr(train_dir):
    print "Building user cvr feature starts."
    if in_memory:
        userID_feature = id_cvr_helper(constants.project_path+"/dataset/raw/user_clean.csv", train_dir, 'userID', 4)
        user_pro_feature = user_profile_cvr(train_dir+"train_with_user_info.csv")
        user_cvr_features = {}
        user_file = open(constants.project_path + "/dataset/raw/clean_id/user.csv", 'r')
        user_file.readline()
        for line in user_file:
            row = line.strip().split(',')
            # if fine_feature:
            feature_list = copy.copy(userID_feature[int(row[0])])
            feature_list.extend(user_pro_feature['gender'][int(row[2])])
            # else:
            # feature_list = copy.copy(user_pro_feature['gender'][int(row[2])])
            feature_list.extend(user_pro_feature['education'][int(row[3])])
            feature_list.extend(user_pro_feature['marriageStatus'][int(row[4])])
            feature_list.extend(user_pro_feature['haveBaby'][int(row[5])])
            user_cvr_features[int(row[0])] = feature_list
        del userID_feature, user_pro_feature
        pickle.dump(user_cvr_features,
                    open(train_dir + 'features/user_cvr_features.pkl', 'wb'))
    else:
        user_cvr_features = pickle.load(
            open(train_dir + 'features/user_cvr_features.pkl', 'rb'))
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
    print pos_features
    return pos_features


# 按positionID处理cvr数据
def build_pos_cvr(train_dir):
    print "Building pos cvr feature starts."
    if in_memory:
        positionID_feature = id_cvr_helper(constants.project_path + "/dataset/raw/position.csv", train_dir, 'positionID', 5)
        pos_info_feature = pos_info_cvr(train_dir+"train_with_pos_info.csv")
        pos_file = open(constants.project_path + "/dataset/raw/clean_id/position.csv", 'r')
        pos_file.readline()
        pos_cvr_features = {}
        i = 1
        for line in pos_file:
            row = line.strip().split(',')
            # if fine_feature:
            feature_list = copy.copy(positionID_feature[int(row[0])])
            feature_list.extend(pos_info_feature['sitesetID'][int(row[1])])
            # else:
            # feature_list = copy.copy(pos_info_feature['sitesetID'][int(row[1])])

            feature_list.extend(pos_info_feature['positionType'][int(row[2])])
            pos_cvr_features[int(row[0])] = feature_list
            # print pos_cvr_features[int(row[0])]
        del positionID_feature, pos_info_feature
        pickle.dump(pos_cvr_features,
                    open(train_dir + 'features/pos_cvr_features.pkl', 'wb'))
    else:
        pos_cvr_features = pickle.load(
                open(train_dir + 'features/pos_cvr_features.pkl', 'rb'))
    return pos_cvr_features


'''
处理广告相关cvr数据
'''


def calibrate(map, key):
    if map[key][1] != 0:
        map[key][1] = round(math.log(map[key][1], 2), 5)
    if map[key][0] != 0:
        map[key][0] = round(math.log10(map[key][0]), 5)
    if key == 0:
        map[key][0] = map[key][1] = 0
    map[key][2] = round((float(map[key][1]) + alpha) / (float(map[key][0]) + beta + alpha), 5)


def build_ad_cvr(train_dir):
    print "Building ad cvr feature starts."
    '''
    :type train_dir: string train_with_ad.csv所在文件夹
    :rtype: dict{creativeID:[ad_cl,ad_cv,ad_cvr,
                            campaign_cl,campaign_cv,campaign_cvr
                            advertiser_cl,advertisr_cv,advertiser_cvr
                            app_cl,app_cv,app_cvr
                            appPlatform_cl,appPlatform_cv,appPlatform_cvr]}
    :Example: a = getADFeature('train_with_ad.csv')
    '''
    if in_memory:

        # length = len(ad_data)
        creative = {}
        ad = {}
        campaign = {}
        advertiser = {}
        app = {}
        appPlatform = {}
        creativeID_adFeature_map = {}
        ad_file_path = train_dir+"train_with_ad_info.csv"
        ad_file = open(ad_file_path, 'r')
        ad_file.readline()
        for line in ad_file:
            # alpha = alphas[connectionType]
            # beta = betas[connectionType]

            row = line.strip().split(',')
            # day = int(row[1])
            # if day < left_day * 10000 or day > right_day * 10000:
            #     continue
            connectionType = int(row[6])
            creativeID_key = int(row[3])
            adID_key = int(row[8])
            campaignID_key = int(row[9])
            advertiserID_key = int(row[10])
            appID_key = int(row[11])
            appPlatform_key = int(row[12])

            # 更新creativeID的数据
            if creativeID_key not in creative:
                creative[creativeID_key] = [0, 0, 0, 0]
            if row[1] == '1':
                creative[creativeID_key][0] += 1
                creative[creativeID_key][1] += 1
            else:
                creative[creativeID_key][0] += 1
            creative[creativeID_key][3] += connection_bias[connectionType]

            # 更新adID的数据
            if adID_key not in ad:
                ad[adID_key] = [0, 0, 0, 0]
            if row[1] == '1':
                ad[adID_key][0] += 1
                ad[adID_key][1] += 1
            else:
                ad[adID_key][0] += 1
            ad[adID_key][3] += connection_bias[connectionType]

            # 更新campaignID的数据
            if campaignID_key not in campaign:
                campaign[campaignID_key] = [0, 0, 0, 0]
            if row[1] == '1':
                campaign[campaignID_key][0] += 1
                campaign[campaignID_key][1] += 1
            else:
                campaign[campaignID_key][0] += 1
            campaign[campaignID_key][3] += connection_bias[connectionType]

            # 更新advertiserID的数据
            if advertiserID_key not in advertiser:
                advertiser[advertiserID_key] = [0, 0, 0, 0]
            if row[1] == '1':
                advertiser[advertiserID_key][0] += 1
                advertiser[advertiserID_key][1] += 1
            else:
                advertiser[advertiserID_key][0] += 1
            advertiser[advertiserID_key][3] += connection_bias[connectionType]

            # 更新appID的数据
            if appID_key not in app:
                app[appID_key] = [0, 0, 0, 0]
            if row[1] == '1':
                app[appID_key][0] += 1
                app[appID_key][1] += 1
            else:
                app[appID_key][0] += 1
            app[appID_key][3] += connection_bias[connectionType]

            # 更新appPlatform的数据
            if appPlatform_key not in appPlatform:
                appPlatform[appPlatform_key] = [0, 0, 0]
            if row[1] == '1':
                appPlatform[appPlatform_key][0] += 1
                appPlatform[appPlatform_key][1] += 1
            else:
                appPlatform[appPlatform_key][0] += 1

        for creativeID_key in creative.keys():
            calibrate(creative, creativeID_key)
            # COEC
            creative[creativeID_key][3] = round((creative[creativeID_key][1] / creative[creativeID_key][3]), 5)

        for adID_key in ad.keys():
            calibrate(ad, adID_key)
            # COEC
            ad[adID_key][3] = round((ad[adID_key][1]/ad[adID_key][3]), 5)

        for campaignID_key in campaign:
            calibrate(campaign, campaignID_key)
            campaign[campaignID_key][3] = round((campaign[campaignID_key][1] / campaign[campaignID_key][3]), 5)

        for advertiserID_key in advertiser:
            calibrate(advertiser, advertiserID_key)
            advertiser[advertiserID_key][3] = round((advertiser[advertiserID_key][1] / advertiser[advertiserID_key][3]), 5)

        for appPlatform_key in appPlatform:
            calibrate(appPlatform, appPlatform_key)

        # if fine_feature:
        for appID_key in app:
            calibrate(app, appID_key)
            app[appID_key][3] = round((app[appID_key][1] / app[appID_key][3]), 5)

        bound = 1

        # 获取最终的list
        ad_file = open(ad_file_path, 'r')
        ad_file.readline()
        for line in ad_file:
            row = line.strip().split(',')
            # day = int(row[1])
            # if day < left_day * 10000 or day > right_day * 10000:
            #     continue
            creative_data = creative[int(row[3])][bound:]
            adID_data = ad[int(row[8])][bound:]
            campaignID_data = campaign[int(row[9])][bound:]
            advertiserID_data = advertiser[int(row[10])][bound:]
            creativeData = adID_data + campaignID_data + advertiserID_data

            appID_data = app[int(row[11])][bound:]
            appPlatform_data = appPlatform[int(row[12])][bound:]
            creativeData += appID_data + appPlatform_data + creative_data

            creativeID_adFeature_map[int(row[3])] = creativeData
        # print "Building ad cvr finished."
        del ad, campaign, advertiser, app, appPlatform, creative
        pickle.dump(creativeID_adFeature_map,
                    open(train_dir + 'features/creativeID_adFeature_map.pkl', 'wb'))
    else:
        creativeID_adFeature_map = pickle.load(
                open(train_dir + 'features/creativeID_adFeature_map.pkl', 'rb'))

    return creativeID_adFeature_map


# 获取appID短期cvr特征
def build_short_cvr(train_dir):
    ad_file_path = train_dir + "train_with_ad_info.csv"
    ad_file = open(ad_file_path, 'r')
    ad_file.readline()
    appID_day_map = {}
    # appID_twoDay_map = {}
    for line in ad_file:
        row = line.strip().split(',')
        label = row[0]
        clickTime = int(row[1])
        day = clickTime / 1000000
        appID = int(row[11])
        if appID not in appID_day_map:
            appID_day_map[appID] = {}
        if day not in appID_day_map[appID]:
            appID_day_map[appID][day] = [0, 0]
        if label == '1':
            appID_day_map[appID][day][0] += 1
            appID_day_map[appID][day][1] += 1
        else:
            appID_day_map[appID][day][0] += 1

    # 将20-31天的appID对应的前两天的conversion amount和cvr存成2维dict：appID_twoDay_map[appID][day]
    # ex: a = appID_twoDay_map[appID][day], conversion = a[0],cvr = a[1]
    # for day in range(20, 32):
    #     for appID in appID_day_map.keys():
    #         if appID not in appID_twoDay_map:
    #             appID_twoDay_map[appID] = {}
    #         if day not in appID_twoDay_map[appID]:
    #             appID_twoDay_map[appID][day] = [0, 0]
    #         yesterday = day - 1
    #         clickYes, clickBeforeYes = 0, 0
    #         converYes, converBeforeYes = 0, 0
    #         if appID in appID_day_map and yesterday in appID_day_map[appID]:
    #             clickYes, converYes = appID_day_map[appID][yesterday][0], appID_day_map[appID][yesterday][1]
    #         # if appID in appID_day_map and beforeYesterday in appID_day_map[appID]:
    #         #     clickBeforeYes, converBeforeYes = appID_day_map[appID][beforeYesterday][0], \
    #         #                                       appID_day_map[appID][beforeYesterday][1]
    #
    #         if clickYes == 0 and not_smooth:
    #             cvr_yes = 0
    #         else:
    #             cvr_yes = round((alpha + converYes) / (float(clickYes) + alpha + beta), 5)
    #         appID_twoDay_map[appID][day] = [round(math.log(converYes, 2), 5), cvr_yes]

    # for k, v in appID_twoDay_map.iteritems():
    #     for day in range(20, 32):
    #         if day in v:
    #             pass
    #             # v[day][0] = round((v[day][0] - min_conversion) / float(max_conversion - min_conversion), 5)
    #             # v[day][0] = round((v[day][0] - min_cvr) / float(max_cvr - min_cvr), 5)
    #         else:
    #             v[day] = [0]
    #             # v[day] = [0, float(alpha)/(alpha+beta)]
    #             # v[day] = [float(alpha) / (alpha + beta)]
    print "Building app short cvr finished."
    return appID_day_map


# build connection feature
def build_conn_cvr(train_dir):
    print "Building connection cvr starts."
    total_df = pd.read_csv(train_dir+"train_with_pos_info.csv")
    # {header:[[3],[3],...,[3]], }
    connection_features = {}
    cvr_helper(total_df, 'connectionType', 5, connection_features)
    cvr_helper(total_df, 'telecomsOperator', 4, connection_features)
    return connection_features


if __name__ == "__main__":
    from features import cvr
    # cvr_handler = cvr.StatisticHandler(constants.custom_path+'/for_train/clean_id/')
    # user_pro_feature = user_profile_cvr(constants.custom_path+'/for_train/clean_id/' + "train_with_user_info.csv")
    # cvr_handler.load_train_cvr()
    # cvr_handler.load_avg_cvr(24, 31)
    ad_map = build_ad_cvr(constants.custom_path+'/for_train/clean_id/')
    # pos_info_cvr(constants.project_path+"/dataset/custom/train_with_pos_info.csv")
    # userID_feature = id_cvr_helper(constants.project_path + "/dataset/raw/user.csv", 'userID')
    # from features.one_hot.user_profile import user_app_feature
    # user_app_cvr = user_app_feature()[0]
    # id_df = pd.DataFrame().from_dict(userID_feature, orient='index')
    # app_df = pd.DataFrame().from_dict(user_app_cvr, orient='index')
    # merge_df = pd.merge(left=app_df, right=id_df)
    # print merge_df