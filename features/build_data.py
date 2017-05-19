# -*- coding:utf-8 -*-
import io

import pandas as pd

import advertising as af
import cvr_test
import gbdt_feature as gf
import position as pf
import user_profile as uf
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
                   user_cvr_features, pos_cvr_features,
                   dataset="train",
                   has_gbdt=False,
                   ffm=False):
    from_file = open(from_path)
    to_file = io.open(to_path, 'w', newline='\n')
    from_file.readline()

    if has_gbdt:
        gbdt_feature = gf.build_gbdt(dataset)

    row_num = 0
    for line in from_file:
        features = {}
        offset = 0
        row = line.strip().split(',')
        field = 0
        # user cvr feature
        user_cvr = user_cvr_features[int(row[4])]
        for i in xrange(len(user_cvr)):
            features[offset+i] = str(field) + "," + str(user_cvr[i]) if ffm else user_cvr[i]
        offset += len(user_cvr)
        field += 1

        # user feature
        user_f = user_features[int(row[4])]
        for i in user_f:
            field += 1
            features[offset+i] = str(field) + "," + str(1) if ffm else 1
        offset += user_dim
        field += 1

        # position feature
        pos_f = pos_features[int(row[5])]
        for i in pos_f:
            features[offset+i] = str(field) + "," + str(1) if ffm else 1
        offset += pos_dim

        site_cvr = pos_cvr_features[int(row[5])][0]
        features[offset+0] = str(field) + "," + str(site_cvr) if ffm else site_cvr

        type_cvr = pos_cvr_features[int(row[5])][1]
        features[offset+1] = str(field) + "," + str(type_cvr) if ffm else type_cvr
        field += 1

        # network feature connection*5, tele-operator*4
        features[offset+int(row[6])] = str(field) + "," + str(1) if ffm else 1
        field += 1
        features[offset+int(row[7])] = str(field) + "," + str(1) if ffm else 1
        offset += 9

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

        if ffm:
            write_as_libffm(features, to_file, row[0])
        else:
            write_as_libfm(features, to_file, row[0])
        if row_num % 100000 == 0:
            print row_num
        row_num += 1
    print "Building x finished."
    from_file.close()
    to_file.close()


def build_x():
    ad_features, ad_dim = af.build_ad_feature(has_id=False)
    user_features, user_dim = uf.build_user_profile()
    pos_features, pos_dim = pf.build_position()
    user_cvr_features = cvr_test.build_user_cvr()
    pos_cvr_features = cvr_test.build_pos_cvr()
    build_x_hepler(constants.local_valid_path, constants.project_path + "/dataset/x_y/split_3/valid_x_pos_no-id.fm",
                   ad_features, ad_dim,
                   user_features, user_dim,
                   pos_features, pos_dim,
                   user_cvr_features, pos_cvr_features,
                   dataset="train",
                   has_gbdt=False,
                   ffm=False)
    build_x_hepler(constants.local_train_path, constants.project_path + "/dataset/x_y/split_3/train_x_pos_no-id.fm",
                   ad_features, ad_dim,
                   user_features, user_dim,
                   pos_features, pos_dim,
                   user_cvr_features, pos_cvr_features,
                   dataset="train",
                   has_gbdt=False,
                   ffm=False)
    build_x_hepler(constants.local_test_path, constants.project_path + "/dataset/x_y/split_3/test_x_pos_no-id.fm",
                   ad_features, ad_dim,
                   user_features, user_dim,
                   pos_features, pos_dim,
                   user_cvr_features, pos_cvr_features,
                   dataset="train",
                   has_gbdt=False,
                   ffm=False)


if __name__ == '__main__':
    build_x()
    build_y(constants.local_train_path, constants.project_path + "/dataset/x_y/split_3/train_y")
    build_y(constants.local_valid_path, constants.project_path + "/dataset/x_y/split_3/valid_y")
    build_y(constants.local_test_path, constants.project_path + "/dataset/x_y/split_3/test_y")
