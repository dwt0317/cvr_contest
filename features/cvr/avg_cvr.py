# -*- coding:utf-8 -*-
import copy
import pandas as pd
from util import constants
import math

'''
用于生成平均cvr特征特征
'''

# alpha = 135  # for smoothing
# beta = 5085

alpha = 123  # for smoothing
beta = 5000

# connectionType cvr
alphas = [9, 225, 22, 1]
betas = [2085, 7246, 2527, 90]

connection_bias = [0.00383, 0.02917, 0.00823, 0.00705, 0.00648]

not_smooth = False
fine_feature = True

left_day = 17
right_day = 24


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
        this_df = total_df[(total_df[header] == category) & (total_df.clickTime >= left_day*10000)
                           & (total_df.clickTime < right_day*10000)]
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
            cv = math.log(cv, 2)
        if click != 0:
            click = math.log10(click)

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


# 统计ID维度的转化率
def id_cvr_helper(raw_file, header, id_index):
    stat = pd.read_csv(raw_file)
    stat_file = open(constants.clean_train_path, 'r')
    id_set = stat[header].values
    del stat
    click_set = {}
    cv_set = {}
    # skip header
    stat_file.readline()
    # 采用按行读取的方式，统计各个id的点击和转化
    for line in stat_file:
        row = line.strip().split(',')
        day = int(row[1])
        if day < left_day * 10000 or day > right_day * 10000:
            continue
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
            cv = math.log(cv, 2)
        if click != 0:
            click = math.log10(click)
        id_cvr[id] = [click, cv, round(cvr, 5)]
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
    userID_feature = id_cvr_helper(constants.project_path+"/dataset/raw/user.csv", 'userID', 4)
    user_pro_feature = user_profile_cvr(train_dir+"train_with_user_info.csv")
    user_cvr_features = {}
    user_file = open(constants.project_path + "/dataset/raw/user.csv", 'r')
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
    print pos_features
    return pos_features


# 按positionID处理cvr数据
def build_pos_cvr(train_dir):
    positionID_feature = id_cvr_helper(constants.project_path + "/dataset/raw/position.csv", 'positionID', 5)
    pos_info_feature = pos_info_cvr(train_dir+"train_with_pos_info.csv")
    pos_file = open(constants.project_path + "/dataset/raw/position.csv", 'r')
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
    return pos_cvr_features


'''
处理广告相关cvr数据
'''


def build_ad_cvr(train_dir):
    '''
    :type train_dir: string train_with_ad.csv所在文件夹
    :rtype: dict{creativeID:[ad_cl,ad_cv,ad_cvr,
                            campaign_cl,campaign_cv,campaign_cvr
                            advertiser_cl,advertisr_cv,advertiser_cvr
                            app_cl,app_cv,app_cvr
                            appPlatform_cl,appPlatform_cv,appPlatform_cvr]}
    :Example: a = getADFeature('train_with_ad.csv')
    '''
    # ad_data = pd.read_csv(filePath)
    ad_file_path = train_dir+"train_with_ad_info.csv"
    ad_file = open(ad_file_path, 'r')
    # length = len(ad_data)
    creative = {}
    ad = {}
    campaign = {}
    advertiser = {}
    app = {}
    appPlatform = {}
    creativeID_adFeature_map = {}
    ad_file.readline()
    for line in ad_file:
        # alpha = alphas[connectionType]
        # beta = betas[connectionType]

        row = line.strip().split(',')
        day = int(row[1])
        if day < left_day * 10000 or day > right_day * 100000:
            continue
        connectionType = int(row[6])
        creativeID_key = int(row[3])
        adID_key = int(row[8])
        campaignID_key = int(row[9])
        advertiserID_key = int(row[10])
        appID_key = int(row[11])
        appPlatform_key = int(row[12])

        # 更新creativeID的数据
        if creativeID_key not in creative:
            creative[creativeID_key] = [0, 0, 0]
        if row[1] == '1':
            creative[creativeID_key][0] += 1
            creative[creativeID_key][1] += 1
        else:
            creative[creativeID_key][0] += 1

        if int(creative[creativeID_key][0]) == 0 and not_smooth:
            creative[creativeID_key][2] = 0
        else:
            creative[creativeID_key][2] = round((float(creative[creativeID_key][1]) + alpha) /
                                          (float(creative[creativeID_key][0]) + beta + alpha), 5)

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
            app[appID_key] = [0, 0, 0]
        if row[1] == '1':
            app[appID_key][0] += 1
            app[appID_key][1] += 1
        else:
            app[appID_key][0] += 1

        # 更新appPlatform的数据
        if appPlatform_key not in appPlatform:
            appPlatform[appPlatform_key] = [0, 0, 0]
        if row[1] == '1':
            appPlatform[appPlatform_key][0] += 1
            appPlatform[appPlatform_key][1] += 1
        else:
            appPlatform[appPlatform_key][0] += 1

    for adID_key in ad.keys():
        if int(ad[adID_key][0]) == 0 and not_smooth:
            ad[adID_key][2] = 0
        else:
            ad[adID_key][2] = round((float(ad[adID_key][1])+alpha) / (float(ad[adID_key][0]) + beta + alpha), 5)
        # COEC
        ad[adID_key][3] = round((ad[adID_key][1]/ad[adID_key][3]), 5)

    for campaignID_key in campaign:
        if float(campaign[campaignID_key][0]) == 0 and not_smooth:
            campaign[campaignID_key][2] = 0
        else:
            campaign[campaignID_key][2] = round((float(campaign[campaignID_key][1]) + alpha) /
                                                (float(campaign[campaignID_key][0]) + beta + alpha), 5)
        campaign[campaignID_key][3] = round((campaign[campaignID_key][1] / campaign[campaignID_key][3]), 5)

    for advertiserID_key in advertiser:
        if float(advertiser[advertiserID_key][0]) == 0 and not_smooth:
            advertiser[advertiserID_key][2] = 0
        else:
            advertiser[advertiserID_key][2] = round((float(advertiser[advertiserID_key][1]) + alpha) / (float(
                advertiser[advertiserID_key][0]) + beta + alpha), 5)
        advertiser[advertiserID_key][3] = round((advertiser[advertiserID_key][1] / advertiser[advertiserID_key][3]), 5)

    for appPlatform_key in appPlatform:
        if float(appPlatform[appPlatform_key][0]) == 0 and not_smooth:
            appPlatform[appPlatform_key][2] = 0
        else:
            appPlatform[appPlatform_key][2] = round((float(appPlatform[appPlatform_key][1]) + alpha) / (float(
                appPlatform[appPlatform_key][0]) + beta + alpha), 5)

    if fine_feature:
        for appID_key in app:
            if float(app[appID_key][0]) == 0 and not_smooth:
                app[appID_key][2] = 0
            else:
                app[appID_key][2] = round((float(app[appID_key][1]) + alpha) / (float(app[appID_key][0]) + beta + alpha), 5)

    bound = 0

    # 获取最终的list
    ad_file = open(ad_file_path, 'r')
    ad_file.readline()
    for line in ad_file:
        row = line.strip().split(',')
        # 只取转化率特征
        creative_data = creative[int(row[3])][bound:]
        adID_data = ad[int(row[8])][bound:]
        campaignID_data = campaign[int(row[9])][bound:]
        advertiserID_data = advertiser[int(row[10])][bound:]
        creativeData = adID_data + campaignID_data + advertiserID_data

        if fine_feature:
            appID_data = app[int(row[11])][bound:]
            appPlatform_data = appPlatform[int(row[12])][bound:]
            creativeData += appID_data + appPlatform_data + creative_data

        creativeID_adFeature_map[int(row[3])] = creativeData
    print "Building ad cvr finished."
    return creativeID_adFeature_map


# 获取appID短期cvr特征
def build_short_cvr(train_dir):
    '''
    :type train_dir: string train_with_ad.csv的路径
    :rtype: dict{appID:{Day20:[conver,cvr]
                        Day21:[conver,cvr]
                            ...
                        Day30:[conver,cvr]
                                }}
    :Example: a = getAppID_cvr_twoDays('train_with_ad_info.csv')
    '''
    ad_file_path = train_dir + "train_with_ad_info.csv"
    ad_file = open(ad_file_path, 'r')
    ad_file.readline()
    appID_day_map = {}
    appID_twoDay_map = {}
    max_conversion = 0
    min_conversion = 1
    max_cvr = 0.0
    min_cvr = 1.0
    for line in ad_file:
        row = line.strip().split(',')
        label = row[0]
        clickTime = row[1]
        day = int(float(clickTime) / 1440.0)
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
    for day in range(20, 32):
        for appID in appID_day_map.keys():
            if appID not in appID_twoDay_map:
                appID_twoDay_map[appID] = {}
            if day not in appID_twoDay_map[appID]:
                appID_twoDay_map[appID][day] = [0, 0]
            yesterday = day - 1
            beforeYesterday = day - 2
            clickYes, clickBeforeYes = 0, 0
            converYes, converBeforeYes = 0, 0
            if appID in appID_day_map and yesterday in appID_day_map[appID]:
                clickYes, converYes = appID_day_map[appID][yesterday][0], appID_day_map[appID][yesterday][1]
            if appID in appID_day_map and beforeYesterday in appID_day_map[appID]:
                clickBeforeYes, converBeforeYes = appID_day_map[appID][beforeYesterday][0], \
                                                  appID_day_map[appID][beforeYesterday][1]

            if clickYes == 0 and not_smooth:
                cvr_yes = 0
            else:
                cvr_yes = (alpha + converYes) / (float(clickYes) + alpha + beta)
            # cvr_before_yes = (alpha + converBeforeYes) / (float(clickBeforeYes) + alpha + beta)
            # avg_conversion = (converYes + converBeforeYes) / 2
            # cvr = (cvr_yes + cvr_before_yes) / 2
            # max_conversion = max(max_conversion, avg_conversion)
            # min_conversion = min(min_conversion, avg_conversion)
            # max_cvr = max(max_cvr, cvr_yes)
            # min_cvr = min(min_cvr, cvr_yes)
            # appID_twoDay_map[appID][day] = [avg_conversion, cvr]
            appID_twoDay_map[appID][day] = [cvr_yes, converYes]

    for k, v in appID_twoDay_map.iteritems():
        for day in range(20, 32):
            if day in v:
                pass
                # v[day][0] = round((v[day][0] - min_conversion) / float(max_conversion - min_conversion), 5)
                # v[day][0] = round((v[day][0] - min_cvr) / float(max_cvr - min_cvr), 5)
            else:
                v[day] = [0]
                # v[day] = [0, float(alpha)/(alpha+beta)]
                # v[day] = [float(alpha) / (alpha + beta)]
    print "Building app short cvr finished."
    return appID_twoDay_map


# build connection feature
def build_conn_cvr(train_dir):
    total_df = pd.read_csv(train_dir+"train_with_pos_info.csv")
    # {header:[[3],[3],...,[3]], }
    connection_features = {}
    cvr_helper(total_df, 'connectionType', 5, connection_features)
    cvr_helper(total_df, 'telecomsOperator', 4, connection_features)
    return connection_features


if __name__ == "__main__":
    pos_info_cvr(constants.project_path+"/dataset/custom/train_with_pos_info.csv")
    # userID_feature = id_cvr_helper(constants.project_path + "/dataset/raw/user.csv", 'userID')
    # from features.one_hot.user_profile import user_app_feature
    # user_app_cvr = user_app_feature()[0]
    # id_df = pd.DataFrame().from_dict(userID_feature, orient='index')
    # app_df = pd.DataFrame().from_dict(user_app_cvr, orient='index')
    # merge_df = pd.merge(left=app_df, right=id_df)
    # print merge_df