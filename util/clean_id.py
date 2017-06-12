# -*- coding:utf-8 -*-
import numpy as np
import constants
import pandas as pd


class IDMapper:
    def __init__(self, idset_dir):
        self._idset_dir = idset_dir
        self._userID_set = set
        self._creativeID_set = set
        # self._adID_set = set
        # self._campaignID_set = set
        self._positionID_set = set
        self.load_idset()

    # load id set
    def load_idset(self):
        self._userID_set = set(np.loadtxt(self._idset_dir+'userID', dtype=int))
        self._creativeID_set = set(np.loadtxt(self._idset_dir + 'creativeID', dtype=int))
        # self._adID_set = set(np.loadtxt(self._idset_dir + 'adID', dtype=int))
        # self._campaignID_set = set(np.loadtxt(self._idset_dir + 'campaignID', dtype=int))
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



def map_func(imp, header, x):
    return imp.get_id(x, header)


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
    train_df.to_csv(to_file, index=False)

if __name__ == '__main__':
    process_id(constants.clean_train_path, constants.custom_path+'/train_clean_id.csv')