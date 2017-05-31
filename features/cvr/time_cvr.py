# -*- coding:utf-8 -*-
import copy
import pandas as pd
from util import constants
import math
import cPickle as pickle
import sys

alpha = 130  # for smoothing
beta = 5085

# alpha = 135  # for smoothing
# beta = 5085

# alpha = 123  # for smoothing
# beta = 5000


'''
用于生成时序特征
'''


# 用户安装行为
def build_user_action():
    # 读取favorite文件
    user_category_file = constants.project_path + "/dataset/custom/favorite/" + "user_app_actions_with_category.csv"
    user_action_dict = {}
    with open(user_category_file, 'r') as f:
        f.readline()
        for line in f:
            row = line.strip().split(',')
            userID = int(row[0])
            day = int(row[1]) / 10000
            category = int(row[3])
            if day in user_action_dict:
                if userID in user_action_dict[day]:
                    user_action_dict[day][userID][category] = user_action_dict[day][userID].get(category, 0) + 1
                else:
                    user_action_dict[day][userID] = {category: 1}
            else:
                user_action_dict[day] = {userID: {category: 1}}
    return user_action_dict


# 30天前用户app类别特征
def build_user_before_action():
    user_before_category_file = constants.project_path + \
                                "/dataset/custom/favorite/user_installedapps_with_category_group.csv"
    user_before_action_dict = {}
    with open(user_before_category_file, 'r') as f:
        f.readline()
        for line in f:
            row = line.strip().split(',')
            userID = int(row[0])
            category = int(row[1])
            number = int(row[2])
            if userID in user_before_action_dict:
                user_before_action_dict[userID][category] = number
            else:
                user_before_action_dict[userID] = {category: number}
    print "Building user before app favorite finished."
    return user_before_action_dict


# 历史转化率信息
def init_history_info(train_ad_file, des_dir):
    # ad_data = pd.read_csv(train_ad_file)
    ad_file = open(train_ad_file, 'r')
    ad_file.readline()

    creativeID_day_dict = {}
    userID_day_dict = {}
    positionID_day_dict = {}
    adID_day_dict = {}
    campaignID_day_dict = {}
    advertiserID_day_dict = {}
    appID_day_dict = {}

    all_info_dict = {}
    curDay = 17

    all_info_dict['userID'] = userID_day_dict
    all_info_dict['positionID'] = positionID_day_dict
    all_info_dict['appID'] = appID_day_dict
    all_info_dict['creativeID'] = creativeID_day_dict
    all_info_dict['advertiserID'] = advertiserID_day_dict
    all_info_dict['adID'] = adID_day_dict
    all_info_dict['campaignID'] = campaignID_day_dict
    max_cvr = 0
    min_cvr = 1
    for line in ad_file:
        row = line.strip().split(',')
        label = int(row[0])
        clickTime = int(row[1])

        creativeID = int(row[3])
        userID = int(row[4])
        positionID = int(row[5])
        adID = int(row[8])
        campainID = int(row[9])
        advertiserID = int(row[10])
        appID = int(row[11])

        day = int(clickTime / 1440)
        # update all_info_dict
        if day != curDay:
            all_info_name = str(day - 1) + '.pkl'
            f1 = file(des_dir+all_info_name, 'wb')
            pickle.dump(all_info_dict, f1, True)
            f1.close()
            curDay += 1
            print "Day " + str(day - 1) + " finished."

        # userID
        if userID not in userID_day_dict:
            userID_day_dict[userID] = [0, 0, 0]
        if label == '1':
            userID_day_dict[userID][0] += 1
            userID_day_dict[userID][1] += 1
        else:
            userID_day_dict[userID][0] += 1
        # if userID_day_dict[userID][0] != 0:
        userID_day_dict[userID][2] = round((alpha + float(userID_day_dict[userID][1])) / (float(
            userID_day_dict[userID][0]) + alpha + beta), 5)

        max_cvr = max(userID_day_dict[userID][2], max_cvr)
        min_cvr = min(userID_day_dict[userID][2], min_cvr)

        # positionID
        if positionID not in positionID_day_dict:
            positionID_day_dict[positionID] = [0, 0, 0]
        if label == '1':
            positionID_day_dict[positionID][0] += 1
            positionID_day_dict[positionID][1] += 1
        else:
            positionID_day_dict[positionID][0] += 1
        # if positionID_day_dict[positionID][0] != 0:
        positionID_day_dict[positionID][2] = round((alpha + float(positionID_day_dict[positionID][1])) / (float(
            positionID_day_dict[positionID][0]) + alpha + beta), 5)

        max_cvr = max(positionID_day_dict[positionID][2], max_cvr)
        min_cvr = min(positionID_day_dict[positionID][2], min_cvr)

        # adID
        if adID not in adID_day_dict:
            adID_day_dict[adID] = [0, 0, 0]
        if label == '1':
            adID_day_dict[adID][0] += 1
            adID_day_dict[adID][1] += 1
        else:
            adID_day_dict[adID][0] += 1
        # if adID_day_dict[adID][0] != 0:
        adID_day_dict[adID][2] = round((alpha + float(adID_day_dict[adID][1])) /
                                       (float(adID_day_dict[adID][0]) + alpha + beta), 5)
        max_cvr = max(adID_day_dict[adID][2], max_cvr)
        min_cvr = min(adID_day_dict[adID][2], min_cvr)

        # campainID
        if campainID not in campaignID_day_dict:
            campaignID_day_dict[campainID] = [0, 0, 0]
        if label == '1':
            campaignID_day_dict[campainID][0] += 1
            campaignID_day_dict[campainID][1] += 1
        else:
            campaignID_day_dict[campainID][0] += 1
        # if campaignID_day_dict[campainID][0] != 0:
        campaignID_day_dict[campainID][2] = round((alpha + float(campaignID_day_dict[campainID][1])) / (float(
            campaignID_day_dict[campainID][0]) + alpha + beta), 5)

        max_cvr = max(campaignID_day_dict[campainID][2], max_cvr)
        min_cvr = min(campaignID_day_dict[campainID][2], min_cvr)

        # creativeID
        if creativeID not in creativeID_day_dict:
            creativeID_day_dict[creativeID] = [0, 0, 0]
        if label == '1':
            creativeID_day_dict[creativeID][0] += 1
            creativeID_day_dict[creativeID][1] += 1
        else:
            creativeID_day_dict[creativeID][0] += 1
        # if creativeID_day_dict[creativeID][0] != 0:
        creativeID_day_dict[creativeID][2] = round((alpha + float(creativeID_day_dict[creativeID][1])) / (float(
            creativeID_day_dict[creativeID][0]) + alpha + beta), 5)

        if creativeID_day_dict[positionID][2] > 0.03:
            print creativeID, creativeID_day_dict[creativeID][2]
            sys.exit(0)

        max_cvr = max(creativeID_day_dict[creativeID][2], max_cvr)
        min_cvr = min(creativeID_day_dict[creativeID][2], min_cvr)

        # advertiserID
        if advertiserID not in advertiserID_day_dict:
            advertiserID_day_dict[advertiserID] = [0, 0, 0]
        if label == '1':
            advertiserID_day_dict[advertiserID][0] += 1
            advertiserID_day_dict[advertiserID][1] += 1
        else:
            advertiserID_day_dict[advertiserID][0] += 1
        # if advertiserID_day_dict[advertiserID][0] != 0:
        advertiserID_day_dict[advertiserID][2] = round((alpha + float(advertiserID_day_dict[advertiserID][1])) / (float(
            advertiserID_day_dict[advertiserID][0]) + alpha + beta), 5)

        max_cvr = max(advertiserID_day_dict[advertiserID][2], max_cvr)
        min_cvr = min(advertiserID_day_dict[advertiserID][2], min_cvr)

        # appID
        if appID not in appID_day_dict:
            appID_day_dict[appID] = [0, 0, 0]
        if label == '1':
            appID_day_dict[appID][0] += 1
            appID_day_dict[appID][1] += 1
        else:
            appID_day_dict[appID][0] += 1
        # if appID_day_dict[appID][0] != 0:
        appID_day_dict[appID][2] = round((alpha + float(appID_day_dict[appID][1])) /
                                         (float(appID_day_dict[appID][0]) + alpha + beta), 5)

        max_cvr = max(appID_day_dict[appID][2], max_cvr)
        min_cvr = min(appID_day_dict[appID][2], min_cvr)

    # update last day
    # all_info_dict['userID'] = userID_day_dict
    # all_info_dict['positionID'] = positionID_day_dict
    # all_info_dict['appID'] = appID_day_dict
    # all_info_dict['creativeID'] = creativeID_day_dict
    # all_info_dict['advertiserID'] = advertiserID_day_dict
    # all_info_dict['adID'] = adID_day_dict
    # all_info_dict['campainID'] = campaignID_day_dict
    all_info_name = str(curDay) + '.pkl'
    f1 = file(des_dir+all_info_name, 'wb')
    pickle.dump(all_info_dict, f1, True)
    f1.close()
    curDay += 1

    print max_cvr, min_cvr


def get_history_info(ID_type, day):
    '''
    :type filePath: ID_type,string,ID的类型
                    ID,int,ID号
                    day,int,某一天
    :rtype: ID_history_dict，dict,{Day17:[ck,conv,cvr]
                                   Day18:[ck,conv,cvr]
                                    ...
                                   Day30:[ck,conv,cvr]
                                    }
    '''
    all_info_name = str(day) + '.pkl'
    all_info = file(des_dir+all_info_name, 'rb')
    all_info_dict = pickle.load(all_info)
    ID_history_dict = all_info_dict[ID_type]
    # ID_history_dict = {}
    # for i in xrange(17, day + 1):
    #     if i in tmp:
    #         ID_history_dict[i] = tmp[i]
    #     else:
    #         ID_history_dict[i] = [0, 0, 0]
    return ID_history_dict


if __name__ == '__main__':
    des_dir = constants.project_path+"/dataset/custom/cvr_statistic/"
    train_ad_file = constants.project_path+"/dataset/custom/train_with_ad_info.csv"
    # init_history_info(train_ad_file, des_dir)
    u = get_history_info('creativeID', 28)
    print u[4565]
    # for k in u.keys()[:5]:
    #     print k, u[k]