# -*- coding:utf-8 -*-
import io

import pandas as pd

import features.gbdt_feature as gf
import one_hot.position as pf
import one_hot.user_profile as uf
from features.one_hot import advertising as af
from util import constants
import cvr
import sys

# 以libfm形式存储
def write_as_libfm(features, file_write, label):
    file_write.write(unicode(label+' '))   # write y  libfm
    for col in sorted(features.keys()):
        file_write.write(unicode(str(col) + ":" + str(features[col]) + ' '))
    file_write.write(unicode('\n'))


# 以libffm形式存储
def write_as_libffm(features, file_write, label):
    file_write.write(unicode(label+' '))   # write y  libfm
    for col in sorted(features.keys()):
        vals = features[col].split(',')  # field, val
        file_write.write(unicode(str(vals[0]) + ":" + str(col) + ":" + str(vals[1]) + ' '))
    file_write.write(unicode('\n'))


# 填充cvr特征
def feed_cvr_feature(features, cvr_f, offset, field, ffm):
    for i in xrange(len(cvr_f)):
        features[offset+i] = str(field) + "," + str(cvr_f[i]) if ffm else cvr_f[i]
    offset += len(cvr_f)
    return offset


# 构造label
def build_y(from_path, to_path):
    df = pd.read_csv(from_path)
    y = df['label']
    print y.head(10)
    y.to_csv(path=to_path, index=False, header=False)


# 计算组合特征
def get_combination_feature(connectionType, appPlatform, appCategory, sitesetID,
                      positionType, age, gender, education,
                      marriageStatus, haveBaby, hometown, residence):
    from one_hot import combination
    f_high = combination.build_high_combination(connectionType, appPlatform, appCategory, sitesetID,
                      positionType, gender, education,
                      marriageStatus, haveBaby, age, hometown, residence)
    f_low = combination.build_low_combination(connectionType, appPlatform, appCategory, sitesetID,
                      positionType, gender, education,
                      marriageStatus, haveBaby, age, hometown, residence)
    f_high.extend(f_low)
    return f_high


# 构造训练集
def build_x_hepler(from_path, to_path,
                   ad_features, ad_dim, ad_map,
                   user_features, user_dim, user_map,
                   pos_features, pos_dim,
                   statistic_handler,
                   data_type="train",
                   has_cvr=False,
                   has_gbdt=False,
                   ffm=False):
    from_file = open(from_path)
    to_file = io.open(to_path, 'w', newline='\n')
    from_file.readline()

    if has_gbdt:
        gbdt_feature = gf.build_gbdt(data_type)

    row_num = 0
    for line in from_file:
        features = {}
        offset = 0
        row = line.strip().split(',')
        field = 0

        creativeID = int(row[3])
        userID = int(row[4])
        positionID = int(row[5])
        if data_type == 'train':
            day = int(row[1]) / 10000
        else:
            day = int(row[2]) / 10000

        # user feature
        user_f = user_features[userID]
        for i in xrange(len(user_f)):
            field += 1
            features[offset+user_f[i]] = str(field) + "," + str(1) if ffm else 1
        offset += user_dim

        if has_cvr:
            userID_cvr = statistic_handler.get_id_cvr('userID', userID, day)
            offset = feed_cvr_feature(features, userID_cvr, offset, field, ffm)
            field += 1
            # print "userID_cvr: ", offset

            positionID_cvr = statistic_handler.get_id_cvr('positionID', positionID, day)
            offset = feed_cvr_feature(features, positionID_cvr, offset, field, ffm)
            field += 1

            # print "positionID_cvr: ", offset

            headers = ['creativeID', 'adID', 'appID', 'campaignID', 'advertiserID']
            for h in headers:
                id_cvr = statistic_handler.get_id_cvr(h, creativeID, day)
                offset = feed_cvr_feature(features, id_cvr, offset, field, ffm)
                field += 1

        if has_cvr:
            # user cvr feature
            user_cvr = statistic_handler.get_user_cvr(data_type, userID)
            offset = feed_cvr_feature(features, user_cvr, offset, field, ffm)
            field += 1

        # user actions
        user_actions_f = statistic_handler.get_user_action(userID, creativeID, day)
        offset = feed_cvr_feature(features, user_actions_f, offset, field, ffm)


        user_before_actions_f = statistic_handler.get_user_before_action(userID, creativeID)
        offset = feed_cvr_feature(features, user_before_actions_f, offset, field, ffm)
        field += 1

        # position feature
        pos_f = pos_features[positionID]
        for i in pos_f:
            features[offset+i] = str(field) + "," + str(1) if ffm else 1
        offset += pos_dim

        # position cvr feature
        if has_cvr:
            pos_cvr = statistic_handler.get_pos_cvr(data_type, int(row[5]))
            offset = feed_cvr_feature(features, pos_cvr, offset, field, ffm)
            field += 1

        # network feature connection * 5, tele-operator * 4
        connectionType = int(row[6])
        features[offset+connectionType] = str(field) + "," + str(1) if ffm else 1
        offset += 5
        telecomsOperator = int(row[7])
        features[offset+telecomsOperator] = str(field) + "," + str(1) if ffm else 1
        offset += 4

        # connection, sitesetID (5*3), connection, positionType (5*6)
        sitesetID = pos_features[int(row[5])][0]
        positionType = pos_features[int(row[5])][1]
        # conn_site = connectionType * 3 + sitesetID
        # features[offset + conn_site] = str(field) + "," + str(1) if ffm else 1
        # offset += 15
        # conn_type = connectionType * 6 + positionType
        # features[offset + conn_type] = str(field) + "," + str(1) if ffm else 1
        # offset += 30
        # field += 1

        # network cvr
        if has_cvr:
            conn_cvr = statistic_handler.get_conn_cvr(int(row[6]), int(row[7]))
            features[offset] = conn_cvr[0][0]
            features[offset+1] = conn_cvr[1][0]
            offset += 2
        field += 1

        if has_gbdt:
            # GBDT feature
            for k in gbdt_feature[row_num]:
                features[k + offset] = 1
            offset += 400
            field += 1

        # ad feature
        ad_f = ad_features[creativeID]
        for i in ad_f:
            features[offset+i] = str(field) + "," + str(1) if ffm else 1
        offset += ad_dim

        # print "ad_f: ", offset
        # ad cvr feature
        if has_cvr:
            ad_cvr = statistic_handler.get_ad_cvr(data_type, int(row[3]))
            for i in xrange(len(ad_cvr)):
                features[offset+i] = str(field) + "," + str(ad_cvr[i]) if ffm else ad_cvr[i]
            offset += len(ad_cvr)
        field += 1

        age, gender, education, marriageStatus, haveBaby, hometown, residence = user_map[userID]
        a_n = len(ad_map[creativeID])
        appPlatform, appCategory = ad_map[creativeID][a_n-2], ad_map[creativeID][a_n-1]
        comb_feature = get_combination_feature(connectionType, appPlatform, appCategory, sitesetID,
                      positionType, age, gender, education,
                      marriageStatus, haveBaby, hometown, residence)
        for i in xrange(len(comb_feature)):
            if comb_feature[i] == 1:
                features[offset+i] = str(field) + "," + str(1) if ffm else 1
        offset += len(comb_feature)


        if ffm:
            write_as_libffm(features, to_file, row[0])
        else:
            write_as_libfm(features, to_file, row[0])
        if row_num % 100000 == 0:
            print row_num
        row_num += 1
        del features
    print "Building x finished."
    from_file.close()
    to_file.close()


# build x file
def build_x():
    has_sparse = False
    ad_features, ad_map, ad_dim = af.build_ad_feature(has_sparse=has_sparse)
    user_features, user_map, user_dim = uf.build_user_profile(has_sparse=has_sparse)
    pos_features, pos_dim = pf.build_position(has_sparse=has_sparse)

    # src_dir_path = constants.project_path+"/dataset/custom/split_6/sample/"
    src_dir_path = constants.project_path + "/dataset/custom/split_6/"
    # src_dir_path = constants.project_path + "/dataset/custom/split_online/"
    # des_dir_path = constants.project_path+"/dataset/x_y/split_online/b11/"
    des_dir_path = constants.project_path + "/dataset/x_y/split_6/b1/"
    cus_dir_path = constants.project_path+"/dataset/custom/"
    # 加载cvr特征
    cvr_handler = cvr.StatisticHandler(cus_dir_path)
    # cvr_handler.load_train_cvr()
    cvr_handler.load_avg_cvr(17, 24)
    cvr_handler.load_time_cvr()

    # # generate online test dataset
    # test_des_file = des_dir_path + "test_x_onehot.fm"
    # test_src_file = constants.project_path+"/dataset/custom/test_re-time.csv"
    # build_x_hepler(test_src_file, test_des_file,
    #                ad_features, ad_dim, ad_map,
    #                user_features, user_dim, user_map,
    #                pos_features, pos_dim,
    #                cvr_handler,
    #                data_type="test",
    #                has_gbdt=False,
    #                ffm=False,
    #                has_cvr=True)

    for i in range(0, 1):
        test_des_file =des_dir_path + "test_x_onehot_" + str(i) + ".fm"
        test_src_file = src_dir_path + "test_x_" + str(i)
        build_x_hepler(test_src_file, test_des_file,
                       ad_features, ad_dim, ad_map,
                       user_features, user_dim, user_map,
                       pos_features, pos_dim,
                       cvr_handler,
                       data_type="train",
                       has_gbdt=False,
                       ffm=False,
                       has_cvr=True)

        # train_src_file = src_dir_path + "train_x_" + str(i) + '_sample'
        # train_des_file = des_dir_path + "train_x_onehot_" + str(i) + ".fms"
        train_src_file = src_dir_path + "train_x_" + str(i)
        train_des_file = des_dir_path + "train_x_onehot_" + str(i) + ".fm"
        build_x_hepler(train_src_file, train_des_file,
                       ad_features, ad_dim, ad_map,
                       user_features, user_dim, user_map,
                       pos_features, pos_dim,
                       cvr_handler,
                       data_type="train",
                       has_gbdt=False,
                       ffm=False,
                       has_cvr=True)


if __name__ == '__main__':
    # cvr_handler = cvr.StatisticHandler(constants.project_path+"/dataset/custom/")
    # cvr_handler.load_train_cvr()
    # cvr_handler.load_avg_cvr(17, 24)
    build_x()
    # build_y(constants.local_train_path, constants.project_path + "/dataset/x_y/split_4/b1/train_y")
    # build_y(constants.local_valid_path, constants.project_path + "/dataset/x_y/split_4/b1/valid_y")
    # build_y(constants.local_test_path, constants.project_path + "/dataset/x_y/split_4/b1/test_y")


