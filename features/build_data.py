# -*- coding:utf-8 -*-
import io

import pandas as pd

import cvr
import features.gbdt_feature as gf
import one_hot.position as pf
import one_hot.user_profile as uf
from features.one_hot import advertising as af
from features.utils import attr_mapper
from util import constants


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


# 填充组合cvr特征
def feed_combine_cvr(attr_map, headers, a_header, features, offset, field, ffm, statistic_handler):
    for header in headers:
        combine_cvr = statistic_handler.get_combine_cvr_feature(a_header, header, attr_map[header], attr_map[a_header])
        offset = feed_cvr_feature(features, combine_cvr, offset, field, ffm)
    field += 1
    return offset, field


# 构造训练集
def build_x_hepler(from_path, to_path,
                   ad_features, ad_dim, ad_map,
                   # user_features, user_dim, user_map,
                   pos_features, pos_dim,
                   statistic_handler,
                   time_gap_dict=None,
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
            day = int(row[1]) / 1000000
        else:
            day = int(row[2]) / 1000000

        # time gap feature
        # instance = int(row[8]) if data_type == "train" else row_num
        # if instance in time_gap_dict:
        #     time_gap = time_gap_dict[instance]
        # else:
        #     time_gap = [0] * 3
        # offset = feed_cvr_feature(features, time_gap, offset, field, ffm)
        # field += 1
        # del time_gap

        # instance = int(row[8]) if data_type == "train" else row_num+1
        # if statistic_handler.is_installed(instance, data_type):
        #     features[offset] = 1
        # else:
        #     features[offset] = 0
        # offset += 1

        # user feature, 之前特征为onehot, 现在改为连续值
        user_f = [int(num) for num in row[8:15]]
        for i in xrange(len(user_f)):
            field += 1
            features[offset+user_f[i]] = str(field) + "," + str(1) if ffm else 1
        offset += len(user_f)
        del user_f

        if has_cvr:
            # user cvr feature
            user_cvr = statistic_handler.get_user_cvr(userID)
            offset = feed_cvr_feature(features, user_cvr, offset, field, ffm)
            field += 1
            del user_cvr

        # user actions
        # user_actions_f = statistic_handler.get_user_action(userID, creativeID, day)
        # offset = feed_cvr_feature(features, user_actions_f, offset, field, ffm)
        # del user_actions_f
        #
        # user_before_actions_f = statistic_handler.get_user_before_action(userID, creativeID)
        # offset = feed_cvr_feature(features, user_before_actions_f, offset, field, ffm)
        # field += 1
        # del user_before_actions_f

        # position feature
        pos_f = pos_features[positionID]
        for i in pos_f:
            features[offset+i] = str(field) + "," + str(1) if ffm else 1
        offset += pos_dim
        del pos_f

        # position cvr feature
        if has_cvr:
            pos_cvr = statistic_handler.get_pos_cvr(int(row[5]))
            offset = feed_cvr_feature(features, pos_cvr, offset, field, ffm)
            field += 1
            del pos_cvr

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

        # network cvr
        if has_cvr:
            conn_cvr = statistic_handler.get_conn_cvr(int(row[6]), int(row[7]))
            features[offset] = conn_cvr[0][0]
            features[offset+1] = conn_cvr[1][0]
            offset += 2
            del conn_cvr
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
        del ad_f

        # ad cvr feature
        if has_cvr:
            ad_cvr = statistic_handler.get_ad_cvr(int(row[3]))
            for i in xrange(len(ad_cvr)):
                features[offset+i] = str(field) + "," + str(ad_cvr[i]) if ffm else ad_cvr[i]
            offset += len(ad_cvr)
            del ad_cvr
        field += 1
        user_map = [int(num) for num in row[8:15]]
        attr_map = attr_mapper.build_attr_map(ad_map, user_map, userID, creativeID, positionID, connectionType)

        # headers = ['appID', 'connectionType', 'campaignID', 'adID', 'creativeID', 'age', 'education', 'gender',
        #            'haveBaby', 'marriageStatus', 'residence', 'appCategory']

        age, gender, education, marriageStatus, haveBaby, hometown, residence = user_map
        a_n = len(ad_map[creativeID])
        appPlatform, appCategory, appID = ad_map[creativeID][a_n - 2], ad_map[creativeID][a_n - 1], ad_map[creativeID][
            a_n - 3]
        campaignID, adID = ad_map[creativeID][a_n - 5], ad_map[creativeID][a_n - 6]
        advertiserID = ad_map[creativeID][a_n - 4]

        # comb_feature = get_combination_feature(connectionType, appPlatform, appCategory, sitesetID,
        #               positionType, age, gender, education,
        #               marriageStatus, haveBaby, hometown, residence)
        # for i in xrange(len(comb_feature)):
        #     if comb_feature[i] == 1:
        #         features[offset+i] = str(field) + "," + str(1) if ffm else 1
        # offset += len(comb_feature)
        # del comb_feature


        # headers = ['age', 'gender', 'education', 'connectionType']
        # offset, field = feed_combine_cvr(attr_map, headers, 'appID', features, offset, field, ffm, statistic_handler)
        #
        # headers = ['appID', 'connectionType', 'campaignID', 'adID', 'creativeID', 'age', 'education', 'gender',
        #            'haveBaby', 'marriageStatus']
        # offset, field = feed_combine_cvr(attr_map, headers, 'positionID', features, offset, field, ffm,
        #                                  statistic_handler)

        # triple_combine_feature = statistic_handler.get_triple_cvr_feature('triple', appID, positionID, connectionType)
        # offset = feed_cvr_feature(features, triple_combine_feature, offset, field, ffm)
        # print offset, len(features)

        # print offset, len(features)
        # print offset, len(features)

        # print offset, len(features)
        # headers = ['age', 'haveBaby', 'education']
        # feed_combine_cvr(attr_map, headers, 'connectionType', features, offset, field, ffm, statistic_handler)

        # headers = ['age', 'haveBaby', 'education', 'residence']
        # feed_combine_cvr(attr_map, headers, 'appCategory', features, offset, field, ffm, statistic_handler)

        # headers = ['education', 'marriageStatus', 'creativeID', 'appID', 'connectionType']
        # feed_combine_cvr(attr_map, headers, 'age', features, offset, field, ffm, statistic_handler)
        # del headers


        # del combine_cvr
        field += 1
        # if data_type == 'test':
        #     day -= 1
        # app_short_fea = statistic_handler.get_app_short_cvr(creativeID, day)
        # offset = feed_cvr_feature(features, app_short_fea, offset, field, ffm)
        # del app_short_fea


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
    # user_features, user_map, user_dim = uf.build_user_profile(has_sparse=has_sparse)
    pos_features, pos_dim = pf.build_position(has_sparse=has_sparse)

    src_dir_path = constants.project_path+"/dataset/custom/split_0/sample/"
    # src_dir_path = constants.project_path + "/dataset/custom/split_online/sample/"
    # des_dir_path = constants.project_path+"/dataset/x_y/split_online/b0/"
    des_dir_path = constants.project_path + "/dataset/x_y/split_0/b0/"
    cus_dir_path = constants.project_path+"/dataset/custom/"
    for_path = constants.custom_path+'/for_train/clean_id/'
    # for_path = constants.custom_path + '/for_predict/clean_id/'

    # 加载cvr特征
    cvr_handler = cvr.StatisticHandler(for_path)
    '''注意online test 的区间是不同的 24 31'''
    cvr_handler.load_avg_cvr(17, 24)

    # cvr_handler.load_time_cvr()
    # train_time_gap, predict_time_gap = cvr_handler.load_time_gap_feature(cus_dir_path)
    # train_last_click, predict_last_click = cvr_handler.load_last_click_feature(cus_dir_path)

    # # generate online test dataset
    # test_des_file = des_dir_path + "test_x.fm"
    # test_src_file = constants.custom_path + "/test_with_user_info.csv"
    # build_x_hepler(test_src_file, test_des_file,
    #                ad_features, ad_dim, ad_map,
    #                # user_features, user_dim, user_map,
    #                pos_features, pos_dim,
    #                cvr_handler,
    #                # time_gap_dict=predict_time_gap,
    #                data_type="test",
    #                has_gbdt=False,
    #                ffm=False,
    #                has_cvr=True)

    for i in range(0, 1):
        test_des_file = des_dir_path + "test_x_" + str(i) + ".fm"
        test_src_file = src_dir_path + "test_x_" + str(i)
        train_src_file = src_dir_path + "train_x_" + str(i) + '_sample'
        train_des_file = des_dir_path + "train_x_" + str(i) + ".fms"
        # train_src_file = src_dir_path + "train_x_" + str(i)
        # train_des_file = des_dir_path + "train_x_onehot_" + str(i) + ".fm"
        #

        #
        build_x_hepler(test_src_file, test_des_file,
                       ad_features, ad_dim, ad_map,
                       # user_features, user_dim, user_map,
                       pos_features, pos_dim,
                       cvr_handler,
                       # time_gap_dict=train_time_gap,
                       # last_click_dict=train_last_click,
                       data_type="train",
                       has_gbdt=False,
                       ffm=False,
                       has_cvr=True)

        build_x_hepler(train_src_file, train_des_file,
                       ad_features, ad_dim, ad_map,
                       # user_features, user_dim, user_map,
                       pos_features, pos_dim,
                       cvr_handler,
                       # time_gap_dict=train_time_gap,
                       # last_click_dict=train_last_click,
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


