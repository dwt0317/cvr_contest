# -*- coding:utf-8 -*-
import csv
import pandas as pd
from util import constants
import numpy as np

alpha = 0.0248  # for smoothing
beta = 75


# gender * 3, education * 8, marriage * 4, baby * 7,
def build_user_cvr_file(train_user_file, feature_file):
    user_cvr_dict = {'gender': np.zeros((3, 2)), 'education': np.zeros((8, 2)), 'marriageStatus': np.zeros((4, 2)),
                     'haveBaby': np.zeros((7, 2))}
    userID_cvr_dict = {}
    headers = ['gender', 'education', 'marriageStatus', 'haveBaby']
    feature_fd = open(feature_file, 'w')
    with open(train_user_file, 'r') as f:
        f.readline()
        num = 0
        for row in f:
            features_list = []
            fields = row.strip().split(',')
            # handle userID cvr feature
            userID = int(fields[4])
            if userID not in userID_cvr_dict:
                userID_cvr_dict[userID] = [0, 0]
            userID_cvr_dict[userID][0] += 1
            if int(fields[0]) == 1:
                userID_cvr_dict[userID][1] += 1

            click = userID_cvr_dict[userID][0]
            cv = userID_cvr_dict[userID][1]
            cvr = round((cv + alpha * beta) / (click + beta), 5)
            features_list.append(str(cvr))

            # index of user profile field
            index = 9
            for header in headers:
                user_cvr_dict[header][int(fields[index])][0] += 1
                # 修改转化数
                if int(fields[0]) == 1:
                    user_cvr_dict[header][int(fields[index])][1] += 1
                # 记录cvr
                click = user_cvr_dict[header][int(fields[index])][0]
                cv = user_cvr_dict[header][int(fields[index])][1]
                cvr = round((cv + alpha * beta) / (click + beta), 5)
                features_list.append(str(cvr))
                index += 1
            feature_fd.write(','.join(features_list))
            feature_fd.write('\n')
            if num % 10000 == 0:
                print num
            num += 1
    feature_fd.close()


# site * 3, type * 6
def build_pos_cvr_file(train_pos_file, feature_file):
    pos_cvr_dict = {'sitesetID': np.zeros((3, 2)), 'positionType': np.zeros((6, 2))}
    headers = ['sitesetID', 'positionType']
    feature_fd = open(feature_file, 'w')
    with open(train_pos_file, 'r') as f:
        f.readline()
        num = 0
        for row in f:
            features_list = []
            fields = row.strip().split(',')
            # index of user profile field
            index = 8
            for header in headers:
                pos_cvr_dict[header][int(fields[index])][0] += 1
                # 修改转化数
                if int(fields[0]) == 1:
                    pos_cvr_dict[header][int(fields[index])][1] += 1
                # 记录cvr
                click = pos_cvr_dict[header][int(fields[index])][0]
                cv = pos_cvr_dict[header][int(fields[index])][1]
                cvr = round((cv + alpha * beta) / (click + beta), 5)
                features_list.append(str(cvr))
                index += 1
            feature_fd.write(','.join(features_list))
            feature_fd.write('\n')
            if num % 10000 == 0:
                print num
            num += 1
    feature_fd.close()


def build_ad_cvr_file(train_ad_file, feature_file):
    '''
    :type train_ad_file: string train_with_ad.csv的路径
          outputPathFile: string AD五个属性的cvr,每一条数据只统计之前的历史数据
    :rtype: null
    :Example: ADFeature_ByLine('train_with_ad.csv','output.csv')
    '''
    # ad_data = pd.read_csv(train_ad_file)
    ad_file = open(train_ad_file, 'r')
    csvfile = file(feature_file, 'wb')
    writer = csv.writer(csvfile)
    # writer.writerow(['creativeID', 'adID', 'campaignID', 'advertiserID', 'appID', 'appPlatformID'])

    ad = {}
    campaign = {}
    advertiser = {}
    app = {}
    appPlatform = {}
    # creativeID_adFeature_map = {}
    ad_file.readline()
    for line in ad_file:
        row = line.strip().split(',')

        adID_key = row[8]
        campaignID_key = row[9]
        advertiserID_key = row[10]
        appID_key = row[11]
        appPlatform_key = row[12]
        # creativeID_key = row[3]

        # 更新adID的数据
        if adID_key not in ad:
            ad[adID_key] = [0, 0, 0]
        if row[0] == '1':
            ad[adID_key][0] += 1
            ad[adID_key][1] += 1
        else:
            ad[adID_key][0] += 1
        if ad[adID_key][0] != 0:
            ad[adID_key][2] = round((float(ad[adID_key][1])+alpha * beta) / (float(ad[adID_key][0]) + beta), 5)

        # 更新campaignID的数据
        if campaignID_key not in campaign:
            campaign[campaignID_key] = [0, 0, 0]
        if row[0] == '1':
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
        if row[0] == '1':
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
        if row[0] == '1':
            app[appID_key][0] += 1
            app[appID_key][1] += 1
        else:
            app[appID_key][0] += 1
        if app[appID_key][0] != 0:
            app[appID_key][2] = round((float(app[appID_key][1]) + alpha * beta) / (float(app[appID_key][0]) + beta), 5)

        # 更新appID的数据
        if appPlatform_key not in appPlatform:
            appPlatform[appPlatform_key] = [0, 0, 0]
        if row[0] == '1':
            appPlatform[appPlatform_key][0] += 1
            appPlatform[appPlatform_key][1] += 1
        else:
            appPlatform[appPlatform_key][0] += 1
        if appPlatform[appPlatform_key][0] != 0:
            appPlatform[appPlatform_key][2] = round((float(appPlatform[appPlatform_key][1]) + alpha * beta) / (float(
                appPlatform[appPlatform_key][0]) + beta), 5)

        writeData = [ad[adID_key][2], campaign[campaignID_key][2], advertiser[advertiserID_key][2],
                     app[appID_key][2], appPlatform[appPlatform_key][2]]
        writer.writerow(writeData)
    csvfile.close()

if __name__ == '__main__':
    dir_path = constants.project_path+"/dataset/custom/split_4/b1/"
    # build_user_cvr_file(dir_path+"train_with_user_info.csv", dir_path+"user_cvr_feature")
    # build_pos_cvr_file(dir_path+"train_with_pos_info.csv", dir_path+"pos_cvr_feature")
    build_ad_cvr_file(dir_path+"train_with_ad_info.csv", dir_path+"ad_cvr_feature")



