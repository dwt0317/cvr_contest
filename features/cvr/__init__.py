# -*- coding:utf-8 -*-
import copy
import pandas as pd
from util import constants


class CVRHandler:
    def __init__(self, dir_path):
        self._dir_path = dir_path

        # for test
        self._pos_cvr_features = {}
        self._user_cvr_features = {}
        self._ad_cvr_features = {}
        self._other_cvr_features = {}

        # for train
        self._pos_cvr_fd = None
        self._user_cvr_fd = None
        self._ad_cvr_fd = None
        self._other_cvr_fd = None

    # 读取test文件所需cvr数据
    def load_test_cvr(self):
        import cvr_test
        self._user_cvr_features = cvr_test.build_user_cvr()
        self._pos_cvr_features = cvr_test.build_pos_cvr()
        print "Loading test cvr finished."

    # 读取train文件所需cvr数据
    def load_train_cvr(self):
        self._user_cvr_fd = open(self._dir_path + "user_cvr_feature")
        self._pos_cvr_fd = open(self._dir_path + "pos_cvr_feature")
        print "Loading train cvr finished."

    # 获取user_cvr feature
    def get_user_cvr(self, data_type, userID):
        if data_type == 'train':
            features = self._user_cvr_fd.readline().strip().split(',')
            return features
        else:
            return self._user_cvr_features[userID]

    def get_pos_cvr(self, data_type, positionID):
        if data_type == 'train':
            features = self._pos_cvr_fd.readline().strip().split(',')
            return features
        else:
            return self._pos_cvr_features[positionID]

    def __del__(self):
        if self._ad_cvr_fd is not None:
            self._pos_cvr_fd.close()
            self._ad_cvr_fd.close()
            self._user_cvr_fd.close()
