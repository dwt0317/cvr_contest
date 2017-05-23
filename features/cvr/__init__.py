# -*- coding:utf-8 -*-
import copy
import pandas as pd
from util import constants


class CVRHandler:
    def __init__(self, dir_path):
        self._dir_path = dir_path
        self._raw_path = constants.project_path+"/dataset/raw/"
        # for test
        self._pos_cvr_features = {}
        self._user_cvr_features = {}
        self._ad_cvr_features = {}
        self._other_cvr_features = {}
        self._creative_app_dict = {}
        self._app_short_cvr_features = {}

        # for train
        self._pos_cvr_fd = None
        self._user_cvr_fd = None
        self._ad_cvr_fd = None
        self._other_cvr_fd = None

    # 读取test文件所需cvr数据, test的统计数据只使用对应train的
    def load_test_cvr(self):
        import cvr_test
        self._user_cvr_features = cvr_test.build_user_cvr(self._dir_path)
        self._pos_cvr_features = cvr_test.build_pos_cvr(self._dir_path)
        # self._ad_cvr_features = cvr_test.build_ad_cvr(self._dir_path)
        self._creative_app_dict = build_creative_app_dict(self._raw_path+"ad.csv")
        self._app_short_cvr_features = cvr_test.build_short_cvr(self._dir_path)
        print "Loading test cvr finished."

    # 读取train文件所需cvr数据
    def load_train_cvr(self):
        self._user_cvr_fd = open(self._dir_path + "user_cvr_feature")
        self._pos_cvr_fd = open(self._dir_path + "pos_cvr_feature")
        print "Loading train cvr finished."

    # 获取user_cvr feature
    def get_user_cvr(self, data_type, userID):
        if data_type == 'before':
            try:
                features = self._user_cvr_fd.readline().strip().split(',')
            except:
                raise Exception("File reaches end!")
            return features
        else:
            return self._user_cvr_features[userID]

    # 获取positive cvr feature
    def get_pos_cvr(self, data_type, positionID):
        if data_type == 'before':
            try:
                features = self._pos_cvr_fd.readline().strip().split(',')
            except:
                raise Exception("File reaches end!")
            return features
        else:
            return self._pos_cvr_features[positionID]

    # 获取ad cvr feature
    def get_ad_cvr(self, data_type, creativeID):
        if data_type == 'before':
            try:
                features = self._ad_cvr_fd.readline().strip().split(',')
            except:
                raise Exception("File reaches end!")
            return features
        else:
            return self._pos_cvr_features[creativeID]

    # 获取app短期表现
    def get_app_short_cvr(self, creativeID, day):
        appID = self._creative_app_dict[creativeID]
        return self._app_short_cvr_features[appID][day]

    def __del__(self):
        if self._ad_cvr_fd is not None:
            self._pos_cvr_fd.close()
            self._ad_cvr_fd.close()
            self._user_cvr_fd.close()


# 建立creative和app的映射关系
def build_creative_app_dict(raw_file):
    app_creative_dict = {}
    with open(raw_file, 'r') as f:
        f.readline()
        for line in f:
            row = line.strip().split(',')
            creativeID = int(row[0])
            appID = int(row[4])
            app_creative_dict[creativeID] = appID

    return app_creative_dict