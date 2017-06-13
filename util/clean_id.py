# -*- coding:utf-8 -*-
import numpy as np
import constants
import pandas as pd


class IDMapper:
    def __init__(self, idset_dir):
        self._idset_dir = idset_dir
        self._userID_set = set
        self._creativeID_set = set
        self._adID_set = set
        self._campaignID_set = set
        self._positionID_set = set
        self.load_idset()

    # load id set
    def load_idset(self):
        self._userID_set = set(np.loadtxt(self._idset_dir+'userID', dtype=int))
        self._creativeID_set = set(np.loadtxt(self._idset_dir + 'creativeID', dtype=int))
        self._adID_set = set(np.loadtxt(self._idset_dir + 'adID', dtype=int))
        self._campaignID_set = set(np.loadtxt(self._idset_dir + 'campaignID', dtype=int))
        self._positionID_set = set(np.loadtxt(self._idset_dir + 'positionID', dtype=int))

    # get id with map
    def get_id(self, id, header):
        if header == 'userID':
            if id in self._userID_set:
                return id
            else:
                return 0
        if header == 'positionID':
            if id in self._positionID_set:
                return id
            else:
                return 0
        if header == 'creativeID':
            if id in self._creativeID_set:
                return id
            else:
                return 0
        if header == 'adID':
            if id in self._adID_set:
                return id
            else:
                return 0
        if header == 'campaignID':
            if id in self._campaignID_set:
                return id
            else:
                return 0



def map_func(imp, header, x):
    return imp.get_id(x, header)


def map_place_func(x):
    if x != 0:
        return x / 100
    return x


# 去掉低频特征
def process_id(raw_file, to_file):
    imp = IDMapper(constants.custom_path+'/idset/')
    train_df = pd.read_csv(raw_file)
    train_df['userID'] = train_df['userID'].apply(lambda x: map_func(imp, 'userID', x))
    print "userID finished."
    train_df['creativeID'] = train_df['creativeID'].apply(lambda x: map_func(imp, 'creativeID', x))
    print "creativeID finished."
    train_df['positionID'] = train_df['positionID'].apply(lambda x: map_func(imp, 'positionID', x))
    print "positionID finished."
    train_df['adID'] = train_df['adID'].apply(lambda x: map_func(imp, 'adID', x))
    print "adID finished."
    train_df['camgaignID'] = train_df['camgaignID'].apply(lambda x: map_func(imp, 'campaignID', x))
    print "campaignID finished."
    train_df.to_csv(to_file, index=False)


# 只保留hometown和residence的省份
def process_place(raw_file, to_file):
    train_df = pd.read_csv(raw_file)
    train_df['hometown'] = train_df['hometown'].apply(lambda x: map_place_func(x))
    train_df['residence'] = train_df['residence'].apply(lambda x: map_place_func(x))
    train_df.to_csv(to_file, index=False)
    print "Place finished."


if __name__ == '__main__':
    # process_id(constants.custom_path+'/for_train/train_with_all_info.csv',
    #            constants.custom_path+'/for_train/clean_id/train_with_all_info.csv')
    # process_id(constants.custom_path + '/for_predict/train_with_ad_info.csv',
    #            constants.custom_path + '/for_predict/clean_id/train_with_ad_info.csv')
    # process_id(constants.custom_path + '/for_predict/train_with_user_info.csv',
    #            constants.custom_path + '/for_predict/clean_id/train_with_user_info.csv')
    # process_id(constants.custom_path + '/for_predict/train.csv',
    #            constants.custom_path + '/for_predict/clean_id/train.csv')
    # process_place(constants.custom_path+'/for_train/clean_id/train_with_all_info.csv',
    #            constants.custom_path+'/for_train/clean_id/train_with_all_info.csv')

    # process_place(constants.custom_path + '/for_train/clean_id/train_with_ad_info.csv',
    #            constants.custom_path + '/for_train/clean_id/train_with_ad_info.csv')

    process_place(constants.custom_path + '/for_predict/clean_id/train_with_user_info.csv',
               constants.custom_path + '/for_predict/clean_id/train_with_user_info.csv')

    # process_place(constants.custom_path + '/for_train/clean_id/train.csv',
    #            constants.custom_path + '/for_train/clean_id/train.csv')
    #
    # process_place(constants.custom_path + '/for_train/clean_id/train_with_pos_info.csv',
    #            constants.custom_path + '/for_train/clean_id/train_with_pos_info.csv')