# -*- coding:utf-8 -*-
import copy
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
def build_pos_cvr_file(pos_user_file, feature_file):
    pos_cvr_dict = {'sitesetID': np.zeros((3, 2)), 'positionType': np.zeros((6, 2))}
    headers = ['sitesetID', 'positionType']
    feature_fd = open(feature_file, 'w')
    with open(pos_user_file, 'r') as f:
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


if __name__ == '__main__':
    dir_path = constants.project_path+"/dataset/custom/split_4/b1/"
    # build_user_cvr_file(dir_path+"train_with_user_info.csv", dir_path+"user_cvr_feature")
    build_pos_cvr_file(dir_path+"train_with_pos_info.csv", dir_path+"pos_cvr_feature")



