# -*- coding:utf-8 -*-
import copy
import pandas as pd
from util import constants
import cPickle as pickle
import math
import sys

alpha = 130  # for smoothing
beta = 5085

# 0.02492 7e-05
max_cvr = 0.025
min_cvr = 7e-05


class StatisticHandler:
    def __init__(self, dir_path):
        self._dir_path = dir_path
        self._raw_path = constants.project_path+"/dataset/raw/"
        # for average
        self._pos_cvr_features = {}
        self._user_cvr_features = {}
        self._ad_cvr_features = {}
        self._other_cvr_features = {}
        self._creative_app_dict = {}
        self._app_short_cvr_features = {}
        self._conn_cvr_features = {}

        # for time
        self._user_action_features = {}
        self._user_before_action_feature = {}
        self._id_cvr_feature = {}
        self._yes_id_cvr_feature = {}
        self._cur_day = 17

        # for train
        self._pos_cvr_fd = None
        self._user_cvr_fd = None
        self._ad_cvr_fd = None
        self._other_cvr_fd = None

        # for mapping
        self._creative_app_dict = build_creative_app_dict(self._raw_path + "ad.csv")
        self._app_category_dict = build_app_category_dict(self._raw_path + "app_categories.csv")

    # 读取平均cvr特征
    def load_avg_cvr(self, left_day, right_day):
        import avg_cvr
        avg_cvr.left_day = left_day
        avg_cvr.right_day = right_day
        self._user_cvr_features = avg_cvr.build_user_cvr(self._dir_path)
        self._pos_cvr_features = avg_cvr.build_pos_cvr(self._dir_path)
        self._ad_cvr_features = avg_cvr.build_ad_cvr(self._dir_path)
        # self._app_short_cvr_features = avg_cvr.build_short_cvr(self._dir_path)
        self._conn_cvr_features = avg_cvr.build_conn_cvr(self._dir_path)
        print "Loading average cvr finished."

    # 读取时序cvr特征
    def load_time_cvr(self):
        import time_cvr
        self._user_action_features = time_cvr.build_user_action()
        self._user_before_action_feature = time_cvr.build_user_before_action()
        # self.load_id_cvr()
        print "Loading time cvr finished."

    def load_id_cvr(self):
        des_dir = constants.project_path + "/dataset/custom/cvr_statistic/"
        if len(self._id_cvr_feature) == 0 and self._cur_day != 17:
            yes_id_cvr_filename = str(self._cur_day-1) + '.pkl'
            yes_all_info = file(des_dir + yes_id_cvr_filename, 'rb')
            self._yes_id_cvr_feature = pickle.load(yes_all_info)
        else:
            del self._yes_id_cvr_feature
        self._yes_id_cvr_feature = self._id_cvr_feature
        id_cvr_filename = str(self._cur_day) + '.pkl'
        all_info = file(des_dir + id_cvr_filename, 'rb')
        self._id_cvr_feature = pickle.load(all_info)

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

    # 获取id类统计信息
    def get_id_cvr(self, header, id, day):
        day -= 1
        if day != self._cur_day:
            self.load_id_cvr()
            self._cur_day = day
            print day
        # creativeID与其他id之间的转化
        if header in ['adID', 'appID', 'campaignID', 'advertiserID']:
            id = self._creative_app_dict[id][header]

        if id in self._id_cvr_feature[header]:
            # !!!!!!!!!!!!!!!!copy
            id_feature = copy.copy(self._id_cvr_feature[header][id])
            if id_feature[0] != 0:
                id_feature[0] = math.log10(id_feature[0])
            if id_feature[1] != 0:
                id_feature[1] = math.log(id_feature[1], 2)
        else:
            id_feature = [0, 0, round(float(alpha)/(alpha+beta), 5)]
        one_day_click = id_feature[0] - self._yes_id_cvr_feature[header].get(id, [0, 0, 0])[0]
        one_day_cv = id_feature[1] - self._yes_id_cvr_feature[header].get(id, [0, 0, 0])[1]
        yes_id_feature = [0, 0]
        if one_day_cv != 0:
            yes_id_feature[0] = math.log(one_day_cv, 2)
        yes_id_feature[1] = round(float(one_day_cv + alpha) / (one_day_click + alpha + beta), 5)
        id_feature.extend(yes_id_feature)
        # numerator = max(id_feature[2]-min_cvr, 0)
        # id_feature[2] = round(numerator / (max_cvr-min_cvr), 5)
        return id_feature[2:]

    # 获取app短期表现(deprecated)
    def get_app_short_cvr(self, creativeID, day):
        appID = self._creative_app_dict[creativeID]['appID']
        return self._app_short_cvr_features[appID][day]

    # 获取user action category喜好
    def get_user_action(self, userID, creativeID, day):
        appID = self._creative_app_dict[creativeID]['appID']
        category = self._app_category_dict[appID]
        d = day - 1
        features = [0, 0, 0, 0, 0]      # diversity, num, most, medium, low
        if d in self._user_action_features:
            if userID in self._user_action_features[d]:
                diversity = len(self._user_action_features[d][userID])
                num = self._user_action_features[d][userID].get(category, 0)
                rate = num / float(diversity)
                features[0], features[1] = diversity, num
                if rate < 0.2:
                    features[4] = 1
                elif rate < 0.6:
                    features[3] = 1
                else:
                    features[2] = 1
        return features

    # 获取30天前用户行为特征
    def get_user_before_action(self, userID, creativeID):
        appID = self._creative_app_dict[creativeID]['appID']
        category = self._app_category_dict[appID]

        features = [0, 0, 0, 0, 0]      # diversity, num, most, medium, low
        if userID in self._user_before_action_feature:
            diversity = len(self._user_before_action_feature[userID])
            num = self._user_before_action_feature[userID].get(category, 0)
            rate = num / float(diversity)
            features[0], features[1] = diversity, num
            if rate < 0.2:
                features[4] = 1
            elif rate < 0.6:
                features[3] = 1
            else:
                features[2] = 1
        return features

    # 获取网络转化率
    def get_conn_cvr(self, connectionType, telecomsOperator):
        conn_cvr = []
        conn_cvr.append(self._conn_cvr_features['connectionType'][connectionType])
        conn_cvr.append(self._conn_cvr_features['telecomsOperator'][telecomsOperator])
        return conn_cvr

    # 获取时间差特征
    def load_time_gap_feature(self, dir_path):
        from features.cvr import time_gap
        # time_gap.build_user_click_time_gap(train_file, test_file, dir_path+"feature/")
        return time_gap.load_user_click_time_gap(dir_path)

    def __del__(self):
        if self._ad_cvr_fd is not None:
            self._pos_cvr_fd.close()
            self._ad_cvr_fd.close()
            self._user_cvr_fd.close()


# 建立creative和ad的映射关系
def build_creative_app_dict(raw_file):
    app_creative_dict = {}
    with open(raw_file, 'r') as f:
        f.readline()
        for line in f:
            row = line.strip().split(',')
            creativeID, adID, campaignID, advertiserID, appID = int(row[0]), int(row[1]), int(row[2]), \
                                                                int(row[3]), int(row[4])
            app_creative_dict[creativeID] = {'adID': adID, 'campaignID': campaignID, 'advertiserID': advertiserID,
                                             'appID': appID}
    return app_creative_dict


def build_app_category_dict(raw_file):
    df = pd.read_csv(raw_file)
    df.set_index('appID', inplace=True)
    return df.to_dict()['appCategory']
