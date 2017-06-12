# -*- coding:utf-8 -*-
import pandas as pd
import constants


# merge user actions with ad info
def install_merge_by_app(raw_file, to_path):
    total_df = pd.read_csv(raw_file)
    ad_df = pd.read_csv(constants.project_path + "/dataset/raw/" + "app_categories.csv")
    merge_df = pd.merge(left=total_df, right=ad_df, how='left', on=['appID'])
    tmp_path = constants.project_path + "/dataset/raw/tmp.action"
    merge_df.to_csv(path_or_buf=tmp_path, index=False)
    #
    user_action_file = tmp_path
    to_f = open(to_path, 'w')
    to_f.write('userID,' + 'appCategory' + '\n')
    # to_f.write(",".join(merge_df.columns.values))
    to_f.write('\n')
    with open(user_action_file, 'r') as f:
        f.readline()
        for line in f:
            row = line.strip().split(',')
            app_cate = int(row[2])
            if app_cate >= 100:
                app_cate /= 100
            row[1] = str(app_cate)
            to_f.write(",".join(row[:2]))
            to_f.write('\n')
    to_f.close()


# 对用户30天前安装数据进行group
def group_user_installedapp():
    train_df = pd.read_csv(constants.clean_train_path)
    userID_set = set(train_df['userID'].unique())
    print len(userID_set)
    header = 'userID,' + 'appCategory,' + 'categoryNum'
    to_file = open(constants.project_path+"/dataset/raw/user_installedapps_with_category_group.csv", 'w')
    to_file.write(header)
    to_file.write('\n')
    with open(constants.project_path+"/dataset/raw/user_installedapps_with_category.csv", 'r') as f:
        f.readlines(2)
        i = 0
        preID = -1
        count_dict = {}
        for line in f:
            row = line.strip().split(',')
            userID = int(row[0])
            if userID == preID:
                count_dict[int(row[1])] = count_dict.get(int(row[1]), 0) + 1
            else:
                for k, v in count_dict.iteritems():
                    to_file.write(str(preID)+','+str(k)+','+str(v))
                    to_file.write('\n')
                count_dict = {}
                count_dict[int(row[1])] = 1
            preID = userID
            if i % 100000 == 0:
                # return
                print i
            i += 1
    to_file.close()