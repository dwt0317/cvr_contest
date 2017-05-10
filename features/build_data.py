# -*- coding:utf-8 -*-
import constants
import pandas as pd
import io
import advertising as af
import position as pf
import user_profile as uf


# 以libfm形式存储
def write_as_libfm(features, file_write, label):
    file_write.write(unicode(label+' '))   # write y  libfm
    for col in features.keys():  # row and column of matrix market start from 1, coo matrix start from 0
        file_write.write(unicode(str(col) + ":" + str(features[col]) + ' '))
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
                   pos_features, pos_dim):
    from_file = open(from_path)
    to_file = io.open(to_path, 'w', newline='\n')
    from_file.readline()

    row = 0
    for line in from_file:
        features = {}
        offset = 0
        fields = line.strip().split(',')

        # user feature
        user_f = user_features[int(fields[4])]
        for i in user_f:
            features[offset+i] = 1
        offset += user_dim

        # position feature
        pos_f = pos_features[int(fields[5])]
        for i in pos_f:
            features[offset+i] = 1
        offset += pos_dim

        # ad feature
        ad_f = ad_features[int(fields[3])]
        for i in ad_f:
            features[offset+i] = 1
        offset += ad_dim

        write_as_libfm(features, to_file, fields[0])
        if row % 100000 == 0:
            print row
        row += 1
    print "Building x finished."
    from_file.close()
    to_file.close()


def build_x():
    ad_features, ad_dim = af.build_ad_feature()
    user_features, user_dim = uf.build_user_profile()
    pos_features, pos_dim = pf.build_position()
    build_x_hepler(constants.local_valid_path, constants.project_path + "/dataset/x_y/local_valid_x",
                   ad_features, ad_dim,
                   user_features, user_dim,
                   pos_features, pos_dim)
    build_x_hepler(constants.local_train_path, constants.project_path + "/dataset/x_y/local_train_x",
                   ad_features, ad_dim,
                   user_features, user_dim,
                   pos_features, pos_dim)
    build_x_hepler(constants.local_test_path, constants.project_path + "/dataset/x_y/local_test_x",
                   ad_features, ad_dim,
                   user_features, user_dim,
                   pos_features, pos_dim)


if __name__ == '__main__':
    build_x()
    # build_y(constants.local_train_path, constants.project_path + "/dataset/custom/local_train_y")
    # build_y(constants.local_valid_path, constants.project_path + "/dataset/custom/local_valid_y")
    # build_y(constants.local_test_path, constants.project_path + "/dataset/custom/local_test_y")