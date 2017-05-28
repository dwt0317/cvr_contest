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
                   cvr_handler,
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

        if has_cvr:
            # user cvr feature
            user_cvr = cvr_handler.get_user_cvr(data_type, int(row[4]))
            for i in xrange(len(user_cvr)):
                features[offset+i] = str(field) + "," + str(user_cvr[i]) if ffm else user_cvr[i]
            offset += len(user_cvr)
            field += 1

        # user feature
        user_f = user_features[int(row[4])]
        for i in xrange(len(user_f)-1):
            field += 1
            features[offset+user_f[i]] = str(field) + "," + str(1) if ffm else 1
        offset += user_dim
        # # 用户安装APP特征 * 1
        # features[offset] = str(field) + "," + str(user_f[len(user_f) - 1]) if ffm else user_f[len(user_f) - 1]
        # offset += 1
        # field += 1

        # position feature
        pos_f = pos_features[int(row[5])]
        for i in pos_f:
            features[offset+i] = str(field) + "," + str(1) if ffm else 1
        offset += pos_dim

        # education, site (8*3), education, type (8*6)
        education = user_features[int(row[4])][2]
        site = pos_features[int(row[5])][0]
        type = pos_features[int(row[5])][1]
        edu_site = education * 3 + site
        features[offset+edu_site] = str(field) + "," + str(1) if ffm else 1
        offset += 24
        edu_type = education * 6 + type
        features[offset + edu_type] = str(field) + "," + str(1) if ffm else 1
        offset += 48
        field += 1

        # if has_cvr:
        # position cvr feature
        pos_cvr = cvr_handler.get_pos_cvr(data_type, int(row[5]))
        for i in xrange(len(pos_cvr)):
            features[offset + i] = str(field) + "," + str(pos_cvr[i]) if ffm else pos_cvr[i]
        offset += len(pos_cvr)
        field += 1

        # network feature connection * 5, tele-operator * 4
        features[offset+int(row[6])] = str(field) + "," + str(1) if ffm else 1
        offset += 5
        features[offset+int(row[7])] = str(field) + "," + str(1) if ffm else 1
        offset += 4
        # network cvr
        if has_cvr:
            conn_cvr = cvr_handler.get_conn_cvr(int(row[6]), int(row[7]))
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
        ad_f = ad_features[int(row[3])]
        for i in ad_f:
            features[offset+i] = str(field) + "," + str(1) if ffm else 1
        offset += ad_dim

        # ad cvr feature
        if has_cvr:
            ad_cvr = cvr_handler.get_ad_cvr(data_type, int(row[3]))
            for i in xrange(len(ad_cvr)):
                features[offset+i] = str(field) + "," + str(ad_cvr[i]) if ffm else ad_cvr[i]
            offset += len(ad_cvr)
            field += 1


        # app short cvr feature
        # if has_cvr:
        #     if data_type == 'train':
        #         day = int((float(row[1]) / 1440.0))
        #     else:
        #         day = int((float(row[2]) / 1440.0))
        #     app_short_cvr = cvr_handler.get_app_short_cvr(int(row[3]), day)
        #     for i in xrange(len(app_short_cvr)):
        #         features[offset+i] = str(field) + "," + str(app_short_cvr[i]) if ffm else app_short_cvr[i]
        #     offset += len(app_short_cvr)
        #     field += 1


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
    ad_features, ad_dim = af.build_ad_feature(has_sparse=False)
    user_features, user_dim = uf.build_user_profile(has_sparse=False)
    pos_features, pos_dim = pf.build_position(has_sparse=False)

    src_dir_path = constants.project_path+"/dataset/custom/split_online/"
    des_dir_path = constants.project_path+"/dataset/x_y/split_online/b9/"
    cus_dir_path = constants.project_path+"/dataset/custom/"
    # 加载cvr特征
    cvr_handler = cvr.CVRHandler(cus_dir_path)
    # cvr_handler.load_train_cvr()
    cvr_handler.load_test_cvr()

    # # generate online test dataset
    test_des_file = des_dir_path + "test_x_onehot.gbdt_online"
    test_src_file = constants.project_path+"/dataset/custom/test_re-time.csv"
    build_x_hepler(test_src_file, test_des_file,
                   ad_features, ad_dim,
                   user_features, user_dim,
                   pos_features, pos_dim,
                   cvr_handler,
                   data_type="test",
                   has_gbdt=False,
                   ffm=False,
                   has_cvr=True)

    for i in range(0, 5):
        # test_des_file = des_dir_path + "test_x_onehot_" + str(i) + ".gbdt_online"
        # test_src_file = src_dir_path + "test_x_" + str(i)
        # build_x_hepler(test_src_file, test_des_file,
        #                ad_features, ad_dim,
        #                user_features, user_dim,
        #                pos_features, pos_dim,
        #                cvr_handler,
        #                data_type="train",
        #                has_gbdt=False,
        #                ffm=False,
        #                has_cvr=True)

        train_src_file = src_dir_path + "train_x_" + str(i)
        train_des_file = des_dir_path + "train_x_onehot_" + str(i) + ".gbdt_online"
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
