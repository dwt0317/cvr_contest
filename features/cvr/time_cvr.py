# -*- coding:utf-8 -*-
import copy
import pandas as pd
from util import constants
import math
import cPickle as pickle
import sys
from features import cvr

alpha = 256  # for smoothing
beta = 9179

# for predict
# left_day = 19
# right_day = 30

# for train
# left_day = 17
# right_day = 28

# for offline
left_day = 17
right_day = 27

# alpha = 135  # for smoothing
# beta = 5085

# alpha = 123  # for smoothing
# beta = 5000


'''
用于生成时序特征
'''


in_memory = False


# 用户安装行为
def build_user_action(src_path):
    if in_memory:
        # 读取favorite文件
        user_category_file = constants.project_path + "/dataset/custom/actions/" + "user_app_actions_with_category.csv"
        user_action_dict = {}   # 用户的安装喜好
        user_install_dict = {}  # 用户是否安装过该app
        with open(user_category_file, 'r') as f:
            f.readline()
            for line in f:
                row = line.strip().split(',')
                userID = int(row[0])
                day = int(row[1]) / 1000000
                if day < left_day or day >= right_day:
                    continue
                category = int(row[3])
                if category >= 100:
                    category /= 100
                if day in user_action_dict:
                    if userID in user_action_dict[day]:
                        user_action_dict[day][userID][category] = user_action_dict[day][userID].get(category, 0) + 1
                    else:
                        user_action_dict[day][userID] = {category: 1}
                else:
                    user_action_dict[day] = {userID: {category: 1}}

                if day in user_install_dict:
                    if userID in user_install_dict[day]:
                        user_install_dict[day][userID] += 1
                    else:
                        user_install_dict[day][userID] = 1
                else:
                    user_install_dict[day] = {userID: 1}

        pickle.dump(user_action_dict,
                    open(constants.custom_path + '/global_features/actions/for_offline/user_action_dict.pkl', 'wb'))
        pickle.dump(user_install_dict,
                    open(constants.custom_path + '/global_features/actions/for_offline/user_install_dict.pkl', 'wb'))
    else:
        # change dir as different task
        user_action_dict = pickle.load(
            open(constants.custom_path + '/global_features/actions/for_offline/user_action_dict.pkl', 'rb'))
        user_install_dict = pickle.load(
            open(constants.custom_path + '/global_features/actions/for_offline/user_install_dict.pkl', 'rb'))
    print "Loading use favorite finished."
    return user_action_dict, user_install_dict


# 30天前用户app类别特征
def build_user_before_action(dir_path):
    if in_memory:
        user_before_category_file = constants.project_path + \
                                    "/dataset/custom/favorite/user_installedapps_with_category_group.csv"
        user_before_action_dict = {}
        user_before_install_dict = {}
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
                if userID in user_before_install_dict:
                    user_before_install_dict[userID] += number
                else:
                    user_before_install_dict[userID] = number
        print "Building user before app favorite finished."
        pickle.dump(user_before_action_dict, open(dir_path + 'features/user_before_action_dict.pkl', 'wb'))
        pickle.dump(user_before_install_dict,
                    open(dir_path + 'features/user_before_install_dict.pkl', 'wb'))
    else:
        user_before_action_dict = pickle.load(open(constants.custom_path + '/features/user_before_action_dict.pkl', 'rb'))
        user_before_install_dict = pickle.load(
            open(constants.custom_path + '/features/user_before_install_dict.pkl', 'rb'))
    return user_before_action_dict, user_before_install_dict


# app安装信息
def build_NDay_installationTimes(nday, dir_path):
    if in_memory:
        action_file = open(constants.raw_path + '/user_app_actions.csv', 'r')
        action_file.readline()
        actions_day_dict = {}
        actions_preNday_dict = {}
        for line in action_file:
            row = line.strip().split(',')
            installTime = row[1]
            appID = row[2]
            day = installTime[0:2]
            if appID not in actions_day_dict:
                actions_day_dict[appID] = {}
                actions_day_dict[appID][day] = 1
            elif day not in actions_day_dict[appID]:
                actions_day_dict[appID][day] = 1
            else:
                actions_day_dict[appID][day] += 1
        action_file.close()

        for appID_key, day_dict_value in actions_day_dict.items():
            if appID_key not in actions_preNday_dict:
                actions_preNday_dict[appID_key] = {}
            for day_key in range(17, 32):
                if day_key not in actions_preNday_dict[appID_key]:
                    actions_preNday_dict[appID_key][day_key] = 0
                for day_key_dummy, installedTimes_value_dummy in day_dict_value.items():
                    if int(day_key_dummy) + nday >= int(day_key) and int(day_key_dummy) < int(day_key):
                        actions_preNday_dict[appID_key][day_key] += installedTimes_value_dummy

        pickle.dump(actions_preNday_dict,
                    open(constants.custom_path + '/global_features/actions/actions_preNday_dict.pkl', 'wb'))
    else:
        actions_preNday_dict = pickle.load(
            open(constants.custom_path + '/global_features/actions/actions_preNday_dict.pkl', 'rb'))
    print "Loading actions pre N day finished."
    return actions_preNday_dict


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
    all_info_name = str(curDay) + '.pkl'
    f1 = file(des_dir+all_info_name, 'wb')
    pickle.dump(all_info_dict, f1, True)
    f1.close()
    curDay += 1

    print max_cvr, min_cvr


if __name__ == '__main__':
    # des_dir = constants.project_path+"/dataset/custom/cvr_statistic/"
    # train_ad_file = constants.project_path+"/dataset/custom/train_with_ad_info.csv"
    # cvr_handler = cvr.StatisticHandler(constants.custom_path + '/for_train/clean_id/')
    # cvr_handler.load_time_cvr()
    build_NDay_installationTimes(10, 'dd')
    build_user_action("dd")
