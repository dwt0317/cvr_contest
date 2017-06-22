# -*- coding:utf-8 -*-
import pandas as pd
import constants
import random


# 全局sample
def total_samplilng(raw_file, to_dir):
    train_df = pd.read_csv(raw_file)
    sample_df = train_df.sample(frac=0.5, random_state=4)
    sample_df.to_csv(to_dir+'/sample_train_05.csv', index=False)

import gc
# 对样本进行负采样
def negative_down_sampling(src_dir_path, rate, l, r):
    for i in range(l, r):
        train_src_file = src_dir_path + "train_x_day_" + str(i)
        train_df = pd.read_csv(train_src_file)
        negative_df = train_df[train_df.label == 0]
        indexes = list(negative_df.index)

        n = len(train_df)
        k = len(indexes)
        train_df.fillna(-1, inplace=True)
        print n, k
        random_list = []
        for m in xrange(int(n * rate)):
            random_list.append(random.randint(0, k - 1))
        print len(random_list)

        positive_list = list(train_df[train_df.label == 1].index)
        random_list.extend(positive_list)
        random_list.sort()
        print random_list[:10]

        train_np = train_df.as_matrix().astype(int)
        # print train_np[:10, :]

        sample_f = open(src_dir_path + 'sample/' + "train_x_day_" + str(i) + '_sample', 'w')
        sample_f.write(','.join(list(train_df.columns.values)))
        sample_f.write('\n')
        for idx in random_list:
            sample_f.write(','.join(str(x) for x in train_np[idx]))
            sample_f.write('\n')
        sample_f.close()
        del train_df, negative_df, positive_list, random_list
        gc.collect()


# bootstrap采样线上的train数据
def bootstrap_online_train(start_date, raw_file, to_dir):
    total_df = pd.read_csv(raw_file)
    total_df['instanceID'] = total_df.index
    partial_df = total_df
    partial_df.fillna(-1, inplace=True)
    data_array = partial_df.as_matrix().astype(int)
    header = list(partial_df.columns.values)
    del total_df, partial_df
    print "Reading data finished."
    for i in xrange(5):
        random.seed()
        train_file = to_dir + "train_x_" + str(i)
        n = len(data_array)
        sample_index = [0] * n
        for j in xrange(n):
            sample_index[j] = random.randint(0, n-1)
        sample_index.sort()
        # np.savetxt(to_dir + "/index/train_x_" + str(i)+'_idx', sample_index, fmt="%s")
        with open(train_file, 'w') as f:
            f.write(','.join(header))
            f.write('\n')
            for j in sample_index:
                f.write(','.join(str(x) for x in data_array[j]))
                f.write('\n')
        print str(i) + " finished."
        del sample_index


# 为online train设置instanceID
def set_instanceID(raw_file, to_file):
    total_df = pd.read_csv(raw_file)
    total_df['instanceID'] = total_df.index
    total_df.to_csv(to_file, index=False)


# 将online test文件切割为小文件
def split_test_file(raw_file, to_dir):
    idx = 0
    to_file = open(to_dir + 'test_' + str(idx), 'w')
    count = 0
    with open(raw_file, 'r') as f:
        for line in f:
            to_file.write(line)
            if count == 500000:
                print idx
                to_file.close()
                idx += 1
                to_file = open(to_dir + 'test_' + str(idx), 'w')
                count = 0
            count += 1


# 丢弃数据集中第30天的样本
def abondon_thirty(raw_file, to_file):
    df = pd.read_csv(raw_file)
    new_df = df[df.clickTime < 30*1000000]
    new_df.to_csv(to_file, index=False)


if __name__ == '__main__':

    # bootstrap_online_train(28, constants.custom_path+'/train_28_29_with_user.csv', constants.custom_path+'/split_online/')
    negative_down_sampling(constants.project_path+"/dataset/custom/split_1/", 0.05, 27, 30)
    # total_samplilng(constants.raw_path, constants.custom_path)
    # split_test_file(constants.project_path+"/dataset/x_y/split_online/b3/test_x.fm",
    #                 constants.project_path + "/dataset/x_y/split_online/b3/test/")
    # set_instanceID(constants.custom_path+'/train_28_29_with_user.csv',
    #                constants.custom_path + '/split_online/train_x_0')
    # for i in xrange(4):
    #     raw_file = constants.custom_path + '/split_0/test_x_' + str(i)
    #     to_file = constants.custom_path + '/split_1/test_x_' + str(i)
    #     abondon_thirty(raw_file, to_file)
    #     print i