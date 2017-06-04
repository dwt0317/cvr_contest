# -*- coding:utf-8 -*-
from util import constants
import pandas as pd
import copy


def remove_installed(test_file, pred):
    user_actions_file = constants.project_path+"/dataset/raw/user_installedapps.csv"
    user_action_df = pd.read_csv(user_actions_file)
    test_df = pd.read_csv(test_file)
    ad_df = pd.read_csv(constants.project_path + "/dataset/raw/" + "ad.csv")

    merge_ad_df = pd.merge(left=test_df, right=ad_df, how='left', on=['creativeID'])
    sample_df = merge_ad_df[['userID', 'appID']]
    merge_install = sample_df.reset_index().merge(user_action_df, on=['userID', 'appID']).set_index('index')
    installed_index = list(merge_install.index)
    tmp = copy.copy(pred)
    for i in installed_index:
        tmp[i] = 0.0
    return tmp

