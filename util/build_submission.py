# -*- coding:utf-8 -*-
import io
import constants
import numpy as np
import pandas as pd


# 规则的hit
def revise_submission(sub_file, train_ad_file, test_file, revise_file):
    user_ad_dict = {}
    with open(train_ad_file, 'r') as f:
        f.readline()
        for line in f:
            row = line.strip().split(',')
            userID = int(row[4])
            appID = int(row[11])
            if int(row[0]) == 1:
                if userID in user_ad_dict:
                    user_ad_dict[userID].add(appID)
                else:
                    user_ad_dict[userID] = set([appID])

    hit_index = []
    with open(test_file, 'r') as f:
        f.readline()
        row_num = 0
        for line in f:
            row = line.strip().split(',')
            userID = int(row[4])
            appID = int(row[11])
            if (userID in user_ad_dict) and (appID in user_ad_dict[userID]):
                hit_index.append(row_num)
            row_num += 1
    print len(hit_index)
    print hit_index[:20]
    sub_list = np.loadtxt(sub_file, skiprows=1, delimiter=',')
    for i in xrange(len(hit_index)):
        sub_list[i][1] = 0.0
    header = "hhh"
    np.savetxt(revise_file, header=header, delimiter=',')


def build_submission(from_path, to_path):
    to_file = io.open(to_path, 'w', newline='\n')
    to_file.write(unicode("lr multi_Cvr all"))
    to_file.write(unicode('\n'))

    with open(from_path) as f:
        instance = 1
        for line in f:
            rcd = str(instance) + "," + line.strip()
            to_file.write(unicode(rcd))
            to_file.write(unicode('\n'))
            instance += 1
    to_file.close()


def check_submission_cvr(sub_file):
    sub_df = pd.read_csv(sub_file, header=None)
    print sub_df.mean()


if __name__ == "__main__":
    # dir_path = constants.project_path+"/dataset/custom/split_5/"
    # revise_submission(constants.project_path+"/submission/ini_sub.csv", dir_path+"train_with_ad_info.csv",
    #                   dir_path+"test_with_ad_info.csv", constants.project_path+"/submission/submission.csv")
    # revise_submission(constants.project_path+"/out/fm_pos_id_no_number.out",
    #                   constants.project_path + "/submission/submission.csv")
    build_submission(constants.project_path + "/out/lr_multi_cvr.out",
                     constants.project_path + "/submission/submission.csv")
    check_submission_cvr(constants.project_path + "/out/lr_multi_cvr.out")
