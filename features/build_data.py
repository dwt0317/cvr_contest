# -*- coding:utf-8 -*-
import io

import pandas as pd

import features.gbdt_feature as gf
import one_hot.position as pf
import one_hot.user_profile as uf
from features.one_hot import advertising as af
from util import constants
import cvr


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


def build_x_hepler(from_path, to_path,
                   ad_features, ad_dim,
                   user_features, user_dim,
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
            day = int((float(row[1]) / 1440.0))
        else:
            day = int((float(row[2]) / 1440.0))

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
                # print h, offset

        if has_cvr:
            # user cvr feature
            user_cvr = statistic_handler.get_user_cvr(data_type, userID)
            offset = feed_cvr_feature(features, user_cvr, offset, field, ffm)
            field += 1
            # print "user_cvr: ", offset
            # for i in xrange(len(user_cvr)):
            #     features[offset+i] = str(field) + "," + str(user_cvr[i]) if ffm else user_cvr[i]
            # offset += len(user_cvr)
            # field += 1

        # user feature
        user_f = user_features[userID]
        for i in xrange(len(user_f)):
            field += 1
            features[offset+user_f[i]] = str(field) + "," + str(1) if ffm else 1
        offset += user_dim
        # print "user_f: ", offset

        # user actions
        user_actions_f = statistic_handler.get_user_action(userID, creativeID, day)
        offset = feed_cvr_feature(features, user_actions_f, offset, field, ffm)
        # for i in xrange(len(user_actions_f)):
        #     features[offset+i] = str(field) + "," + str(user_actions_f[i]) if ffm else user_actions_f[i]
        # offset += len(user_actions_f)
        # print "user_actions_f: ", offset

        user_before_actions_f = statistic_handler.get_user_before_action(userID, creativeID)
        offset = feed_cvr_feature(features, user_before_actions_f, offset, field, ffm)
        # for i in xrange(len(user_before_actions_f)):
        #     features[offset+i] = str(field) + "," + str(user_before_actions_f[i]) if ffm else user_before_actions_f[i]
        # offset += len(user_before_actions_f)
        field += 1
        # print "user_before_actions_f: ", offset
        # position feature
        pos_f = pos_features[positionID]
        for i in pos_f:
            features[offset+i] = str(field) + "," + str(1) if ffm else 1
        offset += pos_dim
        # print "pos_f: ", offset
        # position cvr feature
        if has_cvr:
            pos_cvr = statistic_handler.get_pos_cvr(data_type, int(row[5]))
            offset = feed_cvr_feature(features, pos_cvr, offset, field, ffm)
            # for i in xrange(len(pos_cvr)):
            #     features[offset + i] = str(field) + "," + str(pos_cvr[i]) if ffm else pos_cvr[i]
            # offset += len(pos_cvr)
            field += 1
            # print "pos_cvr: ", offset
        # network feature connection * 5, tele-operator * 4
        connectionType = int(row[6])
        features[offset+connectionType] = str(field) + "," + str(1) if ffm else 1
        offset += 5
        teleOperator = int(row[7])
        features[offset+teleOperator] = str(field) + "," + str(1) if ffm else 1
        offset += 4
        # print "connectionType: ", offset
        # connection, site (5*3), connection, type (5*6)
        site = pos_features[int(row[5])][0]
        type = pos_features[int(row[5])][1]
        conn_site = connectionType * 3 + site
        features[offset + conn_site] = str(field) + "," + str(1) if ffm else 1
        offset += 15
        conn_type = connectionType * 6 + type
        features[offset + conn_type] = str(field) + "," + str(1) if ffm else 1
        offset += 30
        field += 1
        # print "conn_site: ", offset
        # network cvr
        if has_cvr:
            conn_cvr = statistic_handler.get_conn_cvr(int(row[6]), int(row[7]))
            features[offset] = conn_cvr[0][0]
            features[offset+1] = conn_cvr[1][0]
            offset += 2
            # print "conn_cvr: ", offset
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
        # if has_cvr:
        #     ad_cvr = cvr_handler.get_ad_cvr(data_type, int(row[3]))
        #     for i in xrange(len(ad_cvr)):
        #         features[offset+i] = str(field) + "," + str(ad_cvr[i]) if ffm else ad_cvr[i]
        #     offset += len(ad_cvr)
        #     field += 1

        # app short cvr feature, 经gbdt和lr验证确实没用
        # if has_cvr:
        #     app_short_cvr = cvr_handler.get_app_short_cvr(int(row[3]), day)
        #     for i in xrange(len(app_short_cvr)):
        #         features[offset+i] = str(field) + "," + str(app_short_cvr[i]) if ffm else app_short_cvr[i]
        #     offset += len(app_short_cvr)
        #     field += 1
        # return
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

# gbdt特征保留全部cvr特征，不做归一化，用户已安装app数特征
#


def build_x():
    has_sparse = False
    ad_features, ad_dim = af.build_ad_feature(has_sparse=has_sparse)
    user_features, user_dim = uf.build_user_profile(has_sparse=has_sparse)
    pos_features, pos_dim = pf.build_position(has_sparse=has_sparse)

    src_dir_path = constants.project_path+"/dataset/custom/split_5/sample/"
    # src_dir_path = constants.project_path + "/dataset/custom/split_5/"
    # des_dir_path = constants.project_path+"/dataset/x_y/split_5/b12/"
    des_dir_path = constants.project_path + "/dataset/x_y/split_5/b13/"
    cus_dir_path = constants.project_path+"/dataset/custom/"
    # 加载cvr特征
    cvr_handler = cvr.StatisticHandler(cus_dir_path)
    # cvr_handler.load_train_cvr()
    cvr_handler.load_avg_cvr()
    cvr_handler.load_time_cvr()

    # # generate online test dataset
    # test_des_file = des_dir_path + "test_x_onehot.fm"
    # test_src_file = constants.project_path+"/dataset/custom/test_re-time.csv"
    # build_x_hepler(test_src_file, test_des_file,
    #                ad_features, ad_dim,
    #                user_features, user_dim,
    #                pos_features, pos_dim,
    #                cvr_handler,
    #                data_type="test",
    #                has_gbdt=False,
    #                ffm=False,
    #                has_cvr=True)

    for i in range(1, 2):
        test_des_file = des_dir_path + "test_x_onehot_" + str(i) + ".fm"
        test_src_file = src_dir_path + "test_x_" + str(i)
        build_x_hepler(test_src_file, test_des_file,
                       ad_features, ad_dim,
                       user_features, user_dim,
                       pos_features, pos_dim,
                       cvr_handler,
                       data_type="train",
                       has_gbdt=False,
                       ffm=False,
                       has_cvr=True)

        train_src_file = src_dir_path + "train_x_" + str(i) + '_sample'
        train_des_file = des_dir_path + "train_x_onehot_" + str(i) + ".fms"
        # train_src_file = src_dir_path + "train_x_" + str(i)
        # train_des_file = des_dir_path + "train_x_onehot_" + str(i) + ".fm"
        build_x_hepler(train_src_file, train_des_file,
                       ad_features, ad_dim,
                       user_features, user_dim,
                       pos_features, pos_dim,
                       cvr_handler,
                       data_type="train",
                       has_gbdt=False,
                       ffm=False,
                       has_cvr=True)


if __name__ == '__main__':
    build_x()
    # build_y(constants.local_train_path, constants.project_path + "/dataset/x_y/split_4/b1/train_y")
    # build_y(constants.local_valid_path, constants.project_path + "/dataset/x_y/split_4/b1/valid_y")
    # build_y(constants.local_test_path, constants.project_path + "/dataset/x_y/split_4/b1/test_y")


